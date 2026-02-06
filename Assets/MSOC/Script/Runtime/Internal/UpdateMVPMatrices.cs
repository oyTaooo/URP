using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace MSOC
{
    [BurstCompile]
    public struct UpdateMVPMatrices : IJobParallelFor
    {
        public float4x4 vpMatrix;

        public float coverageThreshold;

        [ReadOnly] public NativeArray<bool> cullingResults;

        [ReadOnly] public NativeArray<AABB> bounds;

        [ReadOnly] public NativeArray<float4x4> modelMatrices;

        [WriteOnly] public NativeArray<float4x4> mvpMatrices;

        [WriteOnly] public NativeArray<bool> occluderFlags;

        [WriteOnly] public NativeArray<OccluderSortInfo> occluderInfos;

        public void Execute(int index)
        {
            if (cullingResults[index])
            {
                occluderFlags[index] = false;
                occluderInfos[index] = new OccluderSortInfo { index = index, coverage = 0f, rasterizeTriangleSize = 0 };
                return;
            }

            float coverage = CalcCoverageSIMD(bounds[index], vpMatrix);
            bool isOccluder = coverage >= coverageThreshold;

            occluderFlags[index] = isOccluder;
            occluderInfos[index] = new OccluderSortInfo
            {
                index = index,
                coverage = math.select(0f, coverage, isOccluder),
                rasterizeTriangleSize = 0
            };

            if (isOccluder)
            {
                mvpMatrices[index] = math.mul(vpMatrix, modelMatrices[index]);
            }
        }

        private static float CalcCoverageSIMD(in AABB bounds, in float4x4 vpMatrix)
        {
            float3 center = bounds.center;
            float3 extents = bounds.extents;

            float3 minCorner = center - extents;
            float3 maxCorner = center + extents;

            float4 px0 = new float4(minCorner.x, maxCorner.x, minCorner.x, maxCorner.x);
            float4 py0 = new float4(minCorner.y, minCorner.y, maxCorner.y, maxCorner.y);
            float4 pz0 = new float4(minCorner.z, minCorner.z, minCorner.z, minCorner.z);

            float4 px1 = new float4(minCorner.x, maxCorner.x, minCorner.x, maxCorner.x);
            float4 py1 = new float4(minCorner.y, minCorner.y, maxCorner.y, maxCorner.y);
            float4 pz1 = new float4(maxCorner.z, maxCorner.z, maxCorner.z, maxCorner.z);

            float4 clipX0 = vpMatrix.c0.x * px0 + vpMatrix.c1.x * py0 + vpMatrix.c2.x * pz0 + vpMatrix.c3.x;
            float4 clipY0 = vpMatrix.c0.y * px0 + vpMatrix.c1.y * py0 + vpMatrix.c2.y * pz0 + vpMatrix.c3.y;
            float4 clipZ0 = vpMatrix.c0.z * px0 + vpMatrix.c1.z * py0 + vpMatrix.c2.z * pz0 + vpMatrix.c3.z;
            float4 clipW0 = vpMatrix.c0.w * px0 + vpMatrix.c1.w * py0 + vpMatrix.c2.w * pz0 + vpMatrix.c3.w;

            float4 clipX1 = vpMatrix.c0.x * px1 + vpMatrix.c1.x * py1 + vpMatrix.c2.x * pz1 + vpMatrix.c3.x;
            float4 clipY1 = vpMatrix.c0.y * px1 + vpMatrix.c1.y * py1 + vpMatrix.c2.y * pz1 + vpMatrix.c3.y;
            float4 clipZ1 = vpMatrix.c0.z * px1 + vpMatrix.c1.z * py1 + vpMatrix.c2.z * pz1 + vpMatrix.c3.z;
            float4 clipW1 = vpMatrix.c0.w * px1 + vpMatrix.c1.w * py1 + vpMatrix.c2.w * pz1 + vpMatrix.c3.w;

            const float epsilon = 1e-6f;
            bool4 validW0 = math.abs(clipW0) >= epsilon;
            bool4 validW1 = math.abs(clipW1) >= epsilon;

            float4 invW0 = math.select(1.0f, 1.0f / clipW0, validW0);
            float4 invW1 = math.select(1.0f, 1.0f / clipW1, validW1);

            float4 ndcX0 = clipX0 * invW0;
            float4 ndcY0 = clipY0 * invW0;
            float4 ndcZ0 = clipZ0 * invW0;

            float4 ndcX1 = clipX1 * invW1;
            float4 ndcY1 = clipY1 * invW1;
            float4 ndcZ1 = clipZ1 * invW1;

            bool4 insideFrustum0 = (ndcZ0 >= 0.0f) & (ndcZ0 <= 1.0f) & validW0;
            bool4 insideFrustum1 = (ndcZ1 >= 0.0f) & (ndcZ1 <= 1.0f) & validW1;

            if (!math.any(insideFrustum0) && !math.any(insideFrustum1))
                return 0.0f;


            ndcX0 = math.clamp(ndcX0, -1f, 1f);
            ndcY0 = math.clamp(ndcY0, -1f, 1f);
            ndcX1 = math.clamp(ndcX1, -1f, 1f);
            ndcY1 = math.clamp(ndcY1, -1f, 1f);

            float4 invalidMaxVal = -1000f;
            float4 invalidMinVal = 1000f;

            float4 finalNdcX0 = math.select(invalidMaxVal, ndcX0, validW0);
            float4 finalNdcY0 = math.select(invalidMaxVal, ndcY0, validW0);
            float4 finalNdcX1 = math.select(invalidMaxVal, ndcX1, validW1);
            float4 finalNdcY1 = math.select(invalidMaxVal, ndcY1, validW1);

            float4 finalNdcX0Min = math.select(invalidMinVal, ndcX0, validW0);
            float4 finalNdcY0Min = math.select(invalidMinVal, ndcY0, validW0);
            float4 finalNdcX1Min = math.select(invalidMinVal, ndcX1, validW1);
            float4 finalNdcY1Min = math.select(invalidMinVal, ndcY1, validW1);

            float4 maxX4 = math.max(finalNdcX0, finalNdcX1);
            float4 maxY4 = math.max(finalNdcY0, finalNdcY1);
            float4 minX4 = math.min(finalNdcX0Min, finalNdcX1Min);
            float4 minY4 = math.min(finalNdcY0Min, finalNdcY1Min);

            float maxX = math.cmax(maxX4);
            float maxY = math.cmax(maxY4);
            float minX = math.cmin(minX4);
            float minY = math.cmin(minY4);

            if (minX > maxX || minY > maxY)
                return 0.0f;

            float width = maxX - minX;
            float height = maxY - minY;
            return math.max(width * 0.5f, height * 0.5f);
        }

          
        private static float CalcCoverage(in AABB bounds, in float4x4 vpMatrix)
        {
            float3 center = bounds.center;
            float3 extents = bounds.extents;

            float minX = float.MaxValue;
            float minY = float.MaxValue;
            float maxX = float.MinValue;
            float maxY = float.MinValue;
            for (int i = 0; i < 8; i++)
            {
                float3 offset = new float3(
                    math.select(-extents.x, extents.x, (i & 1) != 0),
                    math.select(-extents.y, extents.y, (i & 2) != 0),
                    math.select(-extents.z, extents.z, (i & 4) != 0)
                    );

                float3 position = center + offset;

                float4 clipSpacePosition = math.mul(vpMatrix, new float4(position, 1.0f));
                if (math.abs(clipSpacePosition.w) < 1e-6f) 
                    continue;

                float3 ndcPosition = clipSpacePosition.xyz / clipSpacePosition.w;
                if (ndcPosition.z < 0.0f || ndcPosition.z > 1.0f) 
                    return 0.0f;

                ndcPosition = math.clamp(ndcPosition, -1f, 1f);
                maxX = math.max(maxX, ndcPosition.x);
                minX = math.min(minX, ndcPosition.x);
                maxY = math.max(maxY, ndcPosition.y);
                minY = math.min(minY, ndcPosition.y);
            }
            if (minX > maxX || minY > maxY) 
                return 0.0f;

            float width = maxX - minX;
            float height = maxY - minY;
            return math.max(width * 0.5f, height * 0.5f);
        }
    }
}
