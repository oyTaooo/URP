using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace MSOC
{
    [BurstCompile]
    public struct BinRasterizer : IJobParallelFor
    {
        public int rasterizeTriangleSize;

        public int depthBufferWidthInTile;

        public int depthBufferHeightInTile;

        public int binWidthInTile;

        public int binHeightInTile;

        public int binWidth;

        public int binHeigh;

        [ReadOnly] public NativeArray<int4> tileRanges;

        [ReadOnly] public NativeArray<float4> depthParams;

        [ReadOnly] public NativeArray<float3> invSlopes;

        [ReadOnly] public NativeArray<int3> vertexX;

        [ReadOnly] public NativeArray<int3> vertexY;

        [ReadOnly] public NativeArray<bool3> isRightEdges;

        [ReadOnly] public NativeArray<int3> flatInfos;

        [NativeDisableParallelForRestriction] public NativeArray<Tile> tiles;


        public void Execute(int index)
        {
            var binIndexX = index % binWidth;
            var binIndexY = index / binWidth;

            var binRange = new int4(
                binIndexX * binWidthInTile,
                binIndexY * binHeightInTile,
                binIndexX == binWidth - 1 ? depthBufferWidthInTile - 1 : (binIndexX + 1) * binWidthInTile - 1,
                binIndexY == binHeigh - 1 ? depthBufferHeightInTile - 1 : (binIndexY + 1) * binHeightInTile - 1
            );

            for (var triangleIdex = 0; triangleIdex < rasterizeTriangleSize; triangleIdex++)
            {
                var tileRange = tileRanges[triangleIdex];
                if (tileRange.x < 0)
                    continue;

                var clampedTileRange = new int4(
                    math.max(tileRange.xy, binRange.xy),
                    math.min(tileRange.zw, binRange.zw)
                );

                if (!math.all(clampedTileRange.xy <= clampedTileRange.zw))
                    continue;

                RasterizeTriangle(triangleIdex, clampedTileRange);
            }
        }

        private void RasterizeTriangle(int triangleIdex, in int4 clampedTileRange)
        {
            var tileRange = tileRanges[triangleIdex];
            var tileRangeOffset = clampedTileRange.xy - tileRange.xy;

            var depthParam = depthParams[triangleIdex];

            var zTileDx = depthParam.x * (MSOCConfig.m_TileSize.x / MSOCConfig.mSubTileSize.x);
            var zTileDy = depthParam.y * (MSOCConfig.m_TileSize.y / MSOCConfig.mSubTileSize.y);

            var zTileRow = depthParam.w + tileRangeOffset.x * zTileDx + tileRangeOffset.y * zTileDy;

            var invSlope = invSlopes[triangleIdex];
            var vx = vertexX[triangleIdex];
            var vy = vertexY[triangleIdex];
            var isRightEdge = isRightEdges[triangleIdex];
            var flatInfo = flatInfos[triangleIdex];

            if (flatInfo.x == 0)
            {
                invSlope = new float3(invSlope[2], invSlope[1], invSlope[0]);
                vx = new int3(vx[2], vx[1], vx[0]);
                vy = new int3(vy[2], vy[1], vy[0]);
                isRightEdge = new bool3(isRightEdge[2], isRightEdge[1], isRightEdge[0]);
            }
            else if (flatInfo.x == 1)
            {
                invSlope = new float3(invSlope[0], invSlope[2], invSlope[1]);
                vx = new int3(vx[0], vx[2], vx[1]);
                vy = new int3(vy[0], vy[2], vy[1]);
                isRightEdge = new bool3(isRightEdge[0], isRightEdge[2], isRightEdge[1]);
            }

            var y = new int4(0, 1, 2, 3) + clampedTileRange.y * MSOCConfig.m_TileSize.y;
            var x0 = new float4(y - vy[0]) * invSlope[0] + vx[0];
            var x1 = new float4(y - vy[1]) * invSlope[1] + vx[1];
            var x2 = flatInfo.x == -1 ? new float4(y - vy[2]) * invSlope[2] + vx[2] : new float4();

            var dx0 = MSOCConfig.m_TileSize.y * invSlope[0];
            var dx1 = MSOCConfig.m_TileSize.y * invSlope[1];
            var dx2 = MSOCConfig.m_TileSize.y * invSlope[2];

            var rightMask = math.select(0u, ~0u, isRightEdge);

            var zSubTileDx = depthParam.x;
            var zTriMax = depthParam.z;
            var zOffsets = new float4(0, 1, 2, 3) * zSubTileDx;
            for (var tileY = clampedTileRange.y; tileY <= clampedTileRange.w; tileY++)
            {
                var z = zTileRow + zOffsets;
                var tileX = clampedTileRange.x;
                var tileIdx = tileY * depthBufferWidthInTile + tileX;
                var xStart = tileX * MSOCConfig.m_TileSize.x;

                for (; tileX <= clampedTileRange.z; tileX++)
                {
                    var ix0 = math.clamp(new int4(x0 - xStart), 0, 32);
                    var ix1 = math.clamp(new int4(x1 - xStart), 0, 32);
                    var ix2 = math.clamp(new int4(x2 - xStart), 0, 32);

                    RasterizeTile(
                        tileIdx, y,
                        math.min(z, zTriMax),
                        rightMask,
                        ix0, ix1, ix2,
                        flatInfo
                    );

                    z += zTileDx;
                    tileIdx++;
                    xStart += MSOCConfig.m_TileSize.x;
                }

                zTileRow += zTileDy;
                y += MSOCConfig.m_TileSize.y;
                x0 += dx0;
                x1 += dx1;
                x2 += dx2;
            }
        }

        private void RasterizeTile(
            int tileIdx, in int4 y,
            in float4 zMax,
            in uint3 rightMask,
            in int4 x0, in int4 x1, in int4 x2, // x0x1x2 in [0, 32]
            in int3 flatInfo)
        {
            var bitmask0 = new uint4(
                (uint)(~0u >> (int)x0.x),
                (uint)(~0u >> (int)x0.y),
                (uint)(~0u >> (int)x0.z),
                (uint)(~0u >> (int)x0.w)
            );

            bitmask0 = math.select(bitmask0, 0u, x0 >= 32);
            bitmask0 ^= rightMask[0];

            if (math.all(bitmask0 == 0u))
                return;

            var bitmask1 = new uint4(
                (uint)(~0u >> (int)x1.x),
                (uint)(~0u >> (int)x1.y),
                (uint)(~0u >> (int)x1.z),
                (uint)(~0u >> (int)x1.w)
            );

            bitmask1 = math.select(bitmask1, 0u, x1 >= 32);
            bitmask1 ^= rightMask[1];

            if (math.all(bitmask1 == 0u))
                return;

            var bitmask = bitmask0 & bitmask1;

            if (flatInfo.z == 0)
            {
                var bitmask2 = new uint4(
                    (uint)(~0u >> (int)x2.x),
                    (uint)(~0u >> (int)x2.y),
                    (uint)(~0u >> (int)x2.z),
                    (uint)(~0u >> (int)x2.w)
                );

                bitmask2 = math.select(bitmask2, 0u, x2 >= 32);
                bitmask2 ^= rightMask[2];
                bitmask &= bitmask2;
            }
            else
            {
                var isAboveHorizon = (y - flatInfo.y >= 0);
                var isFlatTop = (flatInfo.z > 0);
                var inTriangle = isAboveHorizon ^ isFlatTop;
                bitmask &= math.select(0u, ~0u, inTriangle);
            }

            if (math.all(bitmask == 0u))
                return;

            bitmask = ShuffleMask(bitmask);

            UpdateTile(tileIdx, bitmask, zMax);
        }

        private static uint4 ShuffleMask(in uint4 bitmask)
        {
            const uint byteMask = 0xFFu;
            var bytes3 = bitmask & byteMask;     
            var bytes2 = (bitmask >> 8) & byteMask;   
            var bytes1 = (bitmask >> 16) & byteMask;  
            var bytes0 = (bitmask >> 24) & byteMask;

            return new uint4(
                bytes0.x | (bytes0.y << 8) | (bytes0.z << 16) | (bytes0.w << 24),
                bytes1.x | (bytes1.y << 8) | (bytes1.z << 16) | (bytes1.w << 24),
                bytes2.x | (bytes2.y << 8) | (bytes2.z << 16) | (bytes2.w << 24), 
                bytes3.x | (bytes3.y << 8) | (bytes3.z << 16) | (bytes3.w << 24) 
            );
        }

        private void UpdateTile(int tileIdx, in uint4 bitmask, in float4 zMax)
        {
            var tile = tiles[tileIdx];
            const uint fullMask = ~0u;
            for (var i = 0; i < 4; i++)
            {
                var oldMask = tile.bitmask[i];
                var oldZ = tile.z[i];
                var newMask = bitmask[i];
                var newZ = zMax[i];

                if (oldMask == fullMask && newZ >= oldZ) 
                    continue;

                if (newMask == fullMask && newZ < oldZ)
                {
                    tile.z[i] = newZ;
                    tile.bitmask[i] = newMask;
                    continue;
                }

                tile.z[i] = math.select(math.max(oldZ, newZ), newZ, oldMask == 0u);
                tile.bitmask[i] = oldMask | newMask;
            }

            tiles[tileIdx] = tile;
        }
    }
}