using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace MSOC
{
    [BurstCompile]
    public struct PrepareRasterizeTriangleDataJob : IJobParallelFor
    {
        public int depthBufferWidthInTile;

        public int depthBufferHeightInTile;

        public float invTileWidth;

        public float invTileHeight;

        [ReadOnly] public NativeArray<float3> screenSpaceVertices;

        [WriteOnly] public NativeArray<int4> tileRanges;

        [WriteOnly] public NativeArray<float4> depthParams;

        [WriteOnly] public NativeArray<float3> invSlopes;

        [WriteOnly] public NativeArray<int3> vertexX;

        [WriteOnly] public NativeArray<int3> vertexY;

        [WriteOnly] public NativeArray<bool3> isRightEdges;

        [WriteOnly] public NativeArray<int3> flatInfos;

        public void Execute(int index)
        {
            int vertexIndex = index * 3;
            var v0 = screenSpaceVertices[vertexIndex];
            var v1 = screenSpaceVertices[vertexIndex + 1];
            var v2 = screenSpaceVertices[vertexIndex + 2];

            var z = new float3(v0.z, v1.z, v2.z);
                
            if (math.any(z > 1.0f) || math.any(z < 0.0f))
            {
                tileRanges[index] = new int4(-1);
                return;
            }

            var iv0 = new int2(math.round(v0.xy));
            var iv1 = new int2(math.round(v1.xy));
            var iv2 = new int2(math.round(v2.xy));

            var area = (iv1.x - iv0.x) * (iv2.y - iv0.y) - (iv2.x - iv0.x) * (iv1.y - iv0.y);

            // ±³ÃæÌÞ³ý
            if (area > 0)
            {
                tileRanges[index] = new int4(-1);
                return;
            }


            var pixelMinX = math.cmin(new float3(v0.x, v1.x, v2.x));
            var pixelMaxX = math.cmax(new float3(v0.x, v1.x, v2.x));
            var pixelMinY = math.cmin(new float3(v0.y, v1.y, v2.y));
            var pixelMaxY = math.cmax(new float3(v0.y, v1.y, v2.y));

            var tileMinX = math.clamp((int)math.floor(pixelMinX * invTileWidth), 0, depthBufferWidthInTile - 1);
            var tileMinY = math.clamp((int)math.floor(pixelMinY * invTileHeight), 0, depthBufferHeightInTile - 1);
            var tileMaxX = math.clamp((int)math.ceil(pixelMaxX * invTileWidth), 0, depthBufferWidthInTile - 1);
            var tileMaxY = math.clamp((int)math.ceil(pixelMaxY * invTileHeight), 0, depthBufferHeightInTile - 1);
            var tileRange = new int4(tileMinX, tileMinY, tileMaxX, tileMaxY);
            tileRanges[index] = tileRange;

            // dy[0] = v0¡úv1, dy[1] = v1¡úv2, dy[2] = v2¡úv0
            var dy = new int3(iv1.y - iv0.y, iv2.y - iv1.y, iv0.y - iv2.y);
            var zeroMask = math.select(0u, 1u, dy == 0);
            if (math.csum(zeroMask) >= 2)
            {
                tileRanges[index] = new int4(-1);
                return;
            }

            var vy = new int3(iv0.y, iv1.y, iv2.y);
            var vx = new int3(iv0.x, iv1.x, iv2.x);
            var flatInfo = new int3(-1, 0, 0);

            int horizonEdgeIdx = -1;
            if (dy[0] == 0)
            {
                horizonEdgeIdx = 0;
            }
            else if (dy[1] == 0)
            {
                horizonEdgeIdx = 1;
            }
            else if (dy[2] == 0)
            {
                horizonEdgeIdx = 2;
            }

            if (horizonEdgeIdx != -1)
            {
                flatInfo.x = horizonEdgeIdx;
                flatInfo.y = vy[horizonEdgeIdx];
                var nextDy = dy[(horizonEdgeIdx + 1) % 3];
                flatInfo.z = nextDy > 0 ? -1 : 1;
            }

            var isRightEdge = dy < 0;

            var dx = new int3(iv1.x - iv0.x, iv2.x - iv1.x, iv0.x - iv2.x);
            var invSlope = new float3(
                math.select(0f, dx[0] * math.rcp((float)dy[0]), dy[0] != 0),
                math.select(0f, dx[1] * math.rcp((float)dy[1]), dy[1] != 0),
                math.select(0f, dx[2] * math.rcp((float)dy[2]), dy[2] != 0)
            );

            invSlopes[index] = invSlope;
            vertexX[index] = vx;
            vertexY[index] = vy;
            isRightEdges[index] = isRightEdge;
            flatInfos[index] = flatInfo;

            //z = z0 + zPixelDx*(x - x0) + zPixelDy*(y - y0)
            ComputeDepthPlane(v0, v1, v2, out var zPixelDx, out var zPixelDy);

            var maxDepth = math.cmax(z);
            var zSubTileDx = zPixelDx * MSOCConfig.mSubTileSize.x;
            var zSubTileDy = zPixelDy * MSOCConfig.mSubTileSize.y;

            var tileMinXFloat = tileMinX * MSOCConfig.m_TileSize.x;
            var tileMinYFloat = tileMinY * MSOCConfig.m_TileSize.y;
            var zTileMinBase = v0.z + zPixelDx * (tileMinXFloat - v0.x) + zPixelDy * (tileMinYFloat - v0.y);

            var zSubTileMax = zTileMinBase + math.max(0f, zSubTileDx) + math.max(0f, zSubTileDy);

            depthParams[index] = new float4(zSubTileDx, zSubTileDy, maxDepth, zSubTileMax);
        }

        [BurstCompile]
        private static void ComputeDepthPlane(in float3 v0, in float3 v1, in float3 v2, out float zPixelDx, out float zPixelDy)
        {
            var x1 = v1.x - v0.x;
            var x2 = v2.x - v0.x;
            var y1 = v1.y - v0.y;
            var y2 = v2.y - v0.y;
            var z1 = v1.z - v0.z;
            var z2 = v2.z - v0.z;
            var denominator = x1 * y2 - y1 * x2;
            var d = math.select(math.rcp(denominator), 0.0f, math.abs(denominator) < 1e-6f);
            zPixelDx = (z1 * y2 - y1 * z2) * d;
            zPixelDy = (x1 * z2 - z1 * x2) * d;
        }
    }
}