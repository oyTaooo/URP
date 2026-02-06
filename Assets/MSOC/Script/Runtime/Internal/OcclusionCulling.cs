using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace MSOC
{
    [BurstCompile]
    public struct OcclusionCulling : IJobParallelFor
    {
        public int occludersSize;

        public int depthBufferWidth;

        public int depthBufferHeight;

        public int tileHeight;

        public float4x4 vpMatrix;

        [ReadOnly] public NativeSlice<AABB> bounds;

        [ReadOnly] public NativeArray<Tile> tiles;

        [ReadOnly] public NativeArray<bool> occluderFlags;

        public NativeSlice<bool> cullingResults;
        public void Execute(int index)
        {
            if (cullingResults[index])
                return;

            if (index < occludersSize && occluderFlags[index])
                return;

            var depthBufferSize = new int2(depthBufferWidth, depthBufferHeight);
            ComputeRectAndClosestDepth(
                depthBufferSize,
                bounds[index],
                vpMatrix,
                out var rect,
                out var closestDepth
            );

            var tileSize = new int2(MSOCConfig.m_TileSize.x, MSOCConfig.m_TileSize.y);

            var tileMin = rect.xz / tileSize;
            var tileMax = rect.yw / tileSize;
            for (var tileY = tileMin.y; tileY <= tileMax.y; tileY++)
            {
                for (var tileX = tileMin.x; tileX <= tileMax.x; tileX++)
                {
                    var tileRect = new int4(
                        tileX * MSOCConfig.m_TileSize.x,
                        (tileX + 1) * MSOCConfig.m_TileSize.x - 1,
                        tileY * MSOCConfig.m_TileSize.y,
                        (tileY + 1) * MSOCConfig.m_TileSize.y - 1
                    );

                    var intersectRect = ComputeRectIntersection(tileRect, rect);

                    intersectRect.xy -= tileRect.x;
                    intersectRect.zw -= tileRect.z;

                    var coverage = ComputeRectCoverage(intersectRect);

                    var bitmask = ShuffleBitmask(coverage);

                    var tileIdx = tileY * tileHeight + tileX;
                    var tile = tiles[tileIdx];

                    var isOccluded = true;
                    for (var i = 0; i < 4; i++)
                    {
                        if (bitmask[i] == 0)
                            continue;

                        if (closestDepth > tile.z[i] && tile.bitmask[i] == (bitmask[i] | tile.bitmask[i]))
                        {
                            continue;
                        }

                        isOccluded = false;
                        break;
                    }

                    if (!isOccluded)
                    {
                        cullingResults[index] = false;
                        return;
                    }
                }

            }
            cullingResults[index] = true;
        }

        private static uint4 ShuffleBitmask(uint4 bitmask)
        {
            //const uint byteMask = 0xFFu;
            //var a = bitmask & byteMask;       
            //var b = (bitmask >> 8) & byteMask;   
            //var c = (bitmask >> 16) & byteMask;  
            //var d = (bitmask >> 24) & byteMask; 

            //return new uint4(
            //    a.x | (a.y << 8) | (a.z << 16) | (a.w << 24),
            //    b.x | (b.y << 8) | (b.z << 16) | (b.w << 24),
            //    c.x | (c.y << 8) | (c.z << 16) | (c.w << 24),
            //    d.x | (d.y << 8) | (d.z << 16) | (d.w << 24)
            //);
            const uint byteMask = 0xFFu;
            var bytes3 = bitmask & byteMask;          // bit[7:0]
            var bytes2 = (bitmask >> 8) & byteMask;   // bit[15:8]
            var bytes1 = (bitmask >> 16) & byteMask;  // bit[23:16]
            var bytes0 = (bitmask >> 24) & byteMask;  // bit[31:24]

            return new uint4(
                bytes0.x | (bytes0.y << 8) | (bytes0.z << 16) | (bytes0.w << 24),
                bytes1.x | (bytes1.y << 8) | (bytes1.z << 16) | (bytes1.w << 24),
                bytes2.x | (bytes2.y << 8) | (bytes2.z << 16) | (bytes2.w << 24),
                bytes3.x | (bytes3.y << 8) | (bytes3.z << 16) | (bytes3.w << 24)
            );
        }

        private static uint4 ComputeRectCoverage(int4 rect)
        {
                
            var bitCount = rect.y - rect.x + 1;       
            var bits = (1UL << bitCount) - 1;
            //var colMask = (uint)((bits << rect.x) & 0xFFFFFFFFU);
            uint leftMask = ~0u >> rect.x;
            uint rightMask = rect.y < 31 ? (~0u >> (rect.y + 1)) : 0u;
            var colMask = leftMask & ~rightMask;
            var colMasks = new uint4(colMask);

            var rows = new int4(0, 1, 2, 3);

            var rowMaskFlags = (rows >= rect.z) & (rows <= rect.w);

            var rowMasks = math.select(0u, 0xFFFFFFFFu, rowMaskFlags);

            var result = colMasks & rowMasks;

            return result;
        }

        private static int4 ComputeRectIntersection(int4 rect1, int4 rect2)
        {
            var rectMin = math.max(rect1.xz, rect2.xz); 
                                                           
            var rectMax = math.min(rect1.yw, rect2.yw); 

            return new int4(rectMin.x, rectMax.x, rectMin.y, rectMax.y);
        }

        private static void ComputeRectAndClosestDepth(
            in int2 depthBufferSize,
            in AABB bounds,
            in float4x4 mvpMatrix,
            out int4 rect,
            out float closestDepth
            )
        {
            var boundsMin = bounds.min;
            var boundsMax = bounds.max;

            var corners = new NativeArray<float4>(8, Allocator.Temp);
            corners[0] = new float4(boundsMin.x, boundsMin.y, boundsMin.z, 1.0f);
            corners[1] = new float4(boundsMax.x, boundsMin.y, boundsMin.z, 1.0f);
            corners[2] = new float4(boundsMin.x, boundsMax.y, boundsMin.z, 1.0f);
            corners[3] = new float4(boundsMax.x, boundsMax.y, boundsMin.z, 1.0f);
            corners[4] = new float4(boundsMin.x, boundsMin.y, boundsMax.z, 1.0f);
            corners[5] = new float4(boundsMax.x, boundsMin.y, boundsMax.z, 1.0f);
            corners[6] = new float4(boundsMin.x, boundsMax.y, boundsMax.z, 1.0f);
            corners[7] = new float4(boundsMax.x, boundsMax.y, boundsMax.z, 1.0f);
            var screenMin = new int2(int.MaxValue, int.MaxValue);  
            var screenMax = new int2(int.MinValue, int.MinValue);
            var screenSize = new int3(depthBufferSize, 1);
            closestDepth = float.MaxValue;
            for (var i = 0; i < 8; i++)
            {
                var clipSpacePoint = math.mul(mvpMatrix, corners[i]);
                var screenSpacePoint = (clipSpacePoint.xyz / clipSpacePoint.w * 0.5f + 0.5f) * screenSize;
                var screenXY = math.clamp(new int2(screenSpacePoint.xy), 0, screenSize.xy - 1);
                screenMin = math.min(screenMin, screenXY);
                screenMax = math.max(screenMax, screenXY);
                closestDepth = math.min(closestDepth, screenSpacePoint.z);
            }
            rect = new int4(screenMin.x, screenMax.x, screenMin.y, screenMax.y);
            corners.Dispose();
        }
    }
}
