using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace MSOC
{
    [BurstCompile]
    public struct UpdateScreenSpaceData : IJobParallelFor
    {
        public int depthBufferWidth;

        public int depthBufferHeight;

        [ReadOnly] public NativeArray<float3> vertices;

        [ReadOnly] public NativeArray<int> verticesOffests;

        [ReadOnly] public NativeArray<int> verticesIndexes;

        [ReadOnly] public NativeArray<int> verticesIndexesOffsets;

        [ReadOnly] public NativeArray<float4x4> mvpMatrices;

        [ReadOnly] public NativeArray<int> offsets;

        [ReadOnly] public NativeArray<OccluderSortInfo> occluderInfos;

        [NativeDisableParallelForRestriction][WriteOnly] public NativeArray<float3> screenSpaceVertices;

        public void Execute(int index)
        {
            OccluderSortInfo occluderInfo = occluderInfos[index];
            int rasterizeTriangleSize = occluderInfo.rasterizeTriangleSize;

            if (rasterizeTriangleSize <= 0)
                return;

            int objectIndex = occluderInfo.index;
            float4x4 mvpMatrix = mvpMatrices[objectIndex];
            int offset = offsets[objectIndex] * 3;
            int verticesOffest = verticesOffests[objectIndex];
            int indexStart = verticesIndexesOffsets[objectIndex];

            int numVertices = rasterizeTriangleSize * 3;
            int indexEnd = indexStart + numVertices;

            int outputIndex = 0;
            for (int i = indexStart; i < indexEnd; i++, outputIndex++)
            {
                int vertexIndex = verticesIndexes[i] + verticesOffest;
                float3 vertex = vertices[vertexIndex];

                float4 clipSpaceVertex = math.mul(mvpMatrix, new float4(vertex, 1.0f));
                    
                if (clipSpaceVertex.w <= 0.0001f)
                {
                    screenSpaceVertices[offset + outputIndex] = new float3(0, 0, -1);
                    continue;
                }
                    
                float invW = 1.0f / clipSpaceVertex.w;
                float3 ndcSpaceVertex = clipSpaceVertex.xyz * invW;
                float3 screenSpaceVertex = ndcSpaceVertex * 0.5f + 0.5f;
                screenSpaceVertex.x *= depthBufferWidth;
                screenSpaceVertex.y *= depthBufferHeight;
                screenSpaceVertices[offset + outputIndex] = screenSpaceVertex;
            }


        }
    }
}