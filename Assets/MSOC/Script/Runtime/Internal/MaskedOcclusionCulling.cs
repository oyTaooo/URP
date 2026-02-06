using System;
using System.Linq;
using Unity.Burst;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine.UIElements;
using UnityEngine.Rendering;


namespace MSOC
{
    public class MaskedOcclusionCulling
    {
        private MSOCSystem m_CullingSystem;

        private NativeArray<Tile> m_Tiles;

        private NativeArray<float3> m_Vertices;

        private NativeArray<int> m_VerticesIndexes;

        private NativeArray<int> m_VerticesOffest;

        private NativeArray<int> m_VerticesIndexesOffest;

        private NativeArray<int> m_OccluderTriangleSize;

        private NativeArray<float4x4> m_ModelMatrices;

        private NativeArray<float4x4> m_MVPMatrices;

        private NativeArray<bool> m_OccluderFlags;

        private NativeArray<OccluderSortInfo> m_OccluderInfos;

        private NativeArray<int> m_Offsets;

        //triangles
        private NativeArray<int> m_RasterizeTriangleSize;

        private NativeArray<float3> m_ScreenSpaceVertices;

        private NativeArray<int4> m_TileRanges;

        private NativeArray<float3> m_InvSlope;

        private NativeArray<int3> m_VertexX;

        private NativeArray<int3> m_VertexY;

        private NativeArray<bool3> m_IsRightEdge;

        private NativeArray<int3> m_FlatInfo;

        private NativeArray<float4> m_DepthParams;

        private int m_OccluderSize = 0;

        private Matrix4x4 m_VpMatrix;

        public int occluderSize { get { return m_OccluderSize; } }

        public MaskedOcclusionCulling(MSOCSystem cullingSystem)
        {
            m_CullingSystem = cullingSystem;
        }

        public void Dispose()
        {
            m_Tiles.Dispose();
            m_Vertices.Dispose();
            m_VerticesIndexes.Dispose();
            m_VerticesOffest.Dispose();
            m_VerticesIndexesOffest.Dispose();
            m_OccluderTriangleSize.Dispose();
            m_ModelMatrices.Dispose();
            m_MVPMatrices.Dispose();
            m_OccluderFlags.Dispose();
            m_OccluderInfos.Dispose(); 
            m_Offsets.Dispose();
            m_RasterizeTriangleSize.Dispose();
            m_ScreenSpaceVertices.Dispose();
            m_TileRanges.Dispose();
            m_InvSlope.Dispose();
            m_VertexX.Dispose();
            m_VertexY.Dispose();
            m_IsRightEdge.Dispose();
            m_FlatInfo.Dispose();
            m_DepthParams.Dispose();
        }

        public void Initialize()
        {
            int tileCols = MSOCConfig.m_DepthBufferSize.x / MSOCConfig.m_TileSize.x;

            int tileRows = MSOCConfig.m_DepthBufferSize.y / MSOCConfig.m_TileSize.y;

            int tileSize = tileCols * tileRows;

            m_Tiles = new NativeArray<Tile>(tileSize, Allocator.Persistent);
            ClearTiles();

            m_OccluderSize = m_CullingSystem.occluderMeshRenders.Length;
            var occluderMeshFilters = m_CullingSystem.occluderMeshRenders.Select(it => it.GetComponent<MeshFilter>()).ToArray();
            var numOccluderVertices = occluderMeshFilters.Sum(it => it.mesh.vertices.Length);
            var numOccluderVerticesIndices = occluderMeshFilters.Sum(it => it.mesh.triangles.Length);

            m_Vertices = new NativeArray<float3>(numOccluderVertices, Allocator.Persistent);
            m_VerticesIndexes = new NativeArray<int>(numOccluderVerticesIndices, Allocator.Persistent);
            m_VerticesOffest = new NativeArray<int>(m_OccluderSize, Allocator.Persistent);
            m_VerticesIndexesOffest = new NativeArray<int>(m_OccluderSize, Allocator.Persistent);
            m_OccluderTriangleSize = new NativeArray<int>(m_OccluderSize, Allocator.Persistent);
            m_ModelMatrices = new NativeArray<float4x4>(m_OccluderSize, Allocator.Persistent);
            m_MVPMatrices = new NativeArray<float4x4>(m_OccluderSize, Allocator.Persistent);
            m_OccluderFlags = new NativeArray<bool>(m_OccluderSize, Allocator.Persistent);
            m_OccluderInfos = new NativeArray<OccluderSortInfo>(m_OccluderSize, Allocator.Persistent);
            m_Offsets = new NativeArray<int>(m_OccluderSize, Allocator.Persistent);
            //Initialize Occluder Info, can bake into scene
            {
                int vertexIndex = 0;
                int vertexIndexIndex = 0;
                int verticesNum = 0;
                int verticesIndicesNum = 0;
                for (int i = 0; i < m_OccluderSize; i++)
                {
                    var meshFilter = occluderMeshFilters[i];
                    var mesh = meshFilter.sharedMesh;
                    if (mesh == null)
                    {
                        Debug.LogError($"[MSOC] Occluder Mesh is null");
                    }
                    foreach (var vertex in mesh.vertices)
                    {
                        m_Vertices[vertexIndex++] = vertex;
                    }

                    foreach (var triangle in mesh.triangles)
                    {
                        m_VerticesIndexes[vertexIndexIndex++] = triangle;
                    }
                    m_VerticesOffest[i] = verticesNum;
                    m_VerticesIndexesOffest[i] = verticesIndicesNum;
                    verticesNum += mesh.vertices.Length;
                    verticesIndicesNum += mesh.triangles.Length;
                    m_OccluderTriangleSize[i] = mesh.triangles.Length / 3;
                    m_ModelMatrices[i] = occluderMeshFilters[i].transform.localToWorldMatrix;
                }

                m_RasterizeTriangleSize = new NativeArray<int>(1, Allocator.Persistent);
                m_ScreenSpaceVertices = new NativeArray<float3>(MSOCConfig.m_MaxRasterizeTriangleSize * 3, Allocator.Persistent);
                m_TileRanges = new NativeArray<int4>(MSOCConfig.m_MaxRasterizeTriangleSize, Allocator.Persistent);
                m_InvSlope = new NativeArray<float3>(MSOCConfig.m_MaxRasterizeTriangleSize, Allocator.Persistent);
                m_VertexX = new NativeArray<int3>(MSOCConfig.m_MaxRasterizeTriangleSize, Allocator.Persistent);
                m_VertexY = new NativeArray<int3>(MSOCConfig.m_MaxRasterizeTriangleSize, Allocator.Persistent);
                m_IsRightEdge = new NativeArray<bool3>(MSOCConfig.m_MaxRasterizeTriangleSize, Allocator.Persistent);
                m_FlatInfo = new NativeArray<int3>(MSOCConfig.m_MaxRasterizeTriangleSize, Allocator.Persistent);
                m_DepthParams = new NativeArray<float4>(MSOCConfig.m_MaxRasterizeTriangleSize, Allocator.Persistent);

            }

        }

        public void Cull()
        {
            UpdateMvpMatrixAndSelectOccluders().Complete();

            Rasterize();

            OcclusionCulling().Complete();
        }

        private JobHandle UpdateMvpMatrixAndSelectOccluders()
        {
            m_VpMatrix = m_CullingSystem.mainCamera.projectionMatrix * m_CullingSystem.mainCamera.worldToCameraMatrix;

            var updateMVPMatrices = new UpdateMVPMatrices
            {
                cullingResults = m_CullingSystem.cullingResults,
                bounds = m_CullingSystem.bounds,
                modelMatrices = m_ModelMatrices,
                vpMatrix = m_VpMatrix,
                coverageThreshold = MSOCConfig.m_CoverageThreshold,
                mvpMatrices = m_MVPMatrices,
                occluderFlags = m_OccluderFlags,
                occluderInfos = m_OccluderInfos,
            };
            return updateMVPMatrices.Schedule(m_OccluderSize, 64);
        }

        private void Rasterize()
        {
            ClearTiles();

            RasterizeOccluders().Complete();
            //RasterizeOccluders();
        }

        private JobHandle RasterizeOccluders()
        {
            var updateRasterizeDataJob = new UpdateRasterizeDataJob
            {
                occluderSize = m_OccluderSize,
                maxRasterizeTriangleSize = MSOCConfig.m_MaxRasterizeTriangleSize,
                occluderTriangleSize = m_OccluderTriangleSize,
                offsets = m_Offsets,
                rasterizeTriangleSize = m_RasterizeTriangleSize,
                occluderInfos = m_OccluderInfos
            };
            updateRasterizeDataJob.Run();

            var updateScreenSpaceDataJob = new UpdateScreenSpaceData
            {
                depthBufferWidth = MSOCConfig.m_DepthBufferSize.x,
                depthBufferHeight = MSOCConfig.m_DepthBufferSize.y,
                vertices = m_Vertices,
                verticesOffests = m_VerticesOffest,
                verticesIndexes = m_VerticesIndexes,
                verticesIndexesOffsets = m_VerticesIndexesOffest,
                mvpMatrices = m_MVPMatrices,
                offsets = m_Offsets,
                occluderInfos = m_OccluderInfos,
                screenSpaceVertices = m_ScreenSpaceVertices
            };
            var updateScreenSpaceDataJobHandle = updateScreenSpaceDataJob.Schedule(m_OccluderSize, 64);

            var prepareRasterizeTriangleDataJob = new PrepareRasterizeTriangleDataJob
            {
                depthBufferWidthInTile = MSOCConfig.m_DepthBufferSize.x / MSOCConfig.m_TileSize.x,
                depthBufferHeightInTile = MSOCConfig.m_DepthBufferSize.y / MSOCConfig.m_TileSize.y,
                invTileWidth = (float)1 / MSOCConfig.m_TileSize.x,
                invTileHeight = (float)1 / MSOCConfig.m_TileSize.y,
                screenSpaceVertices = m_ScreenSpaceVertices,
                tileRanges = m_TileRanges,
                depthParams = m_DepthParams,
                invSlopes = m_InvSlope,
                vertexX = m_VertexX,
                vertexY = m_VertexY,
                isRightEdges = m_IsRightEdge,
                flatInfos = m_FlatInfo
            };
            var rasterizeTriangleSize = m_RasterizeTriangleSize[0];
            var prepareRasterizeTriangleDataJobHandle = prepareRasterizeTriangleDataJob.Schedule(rasterizeTriangleSize, 64, updateScreenSpaceDataJobHandle);
               
            var binRasterizerJob = new BinRasterizer
            {
                rasterizeTriangleSize = rasterizeTriangleSize,
                depthBufferWidthInTile = MSOCConfig.m_DepthBufferSize.x / MSOCConfig.m_TileSize.x,
                depthBufferHeightInTile = MSOCConfig.m_DepthBufferSize.y / MSOCConfig.m_TileSize.y,
                binWidthInTile = MSOCConfig.m_DepthBufferSize.x / MSOCConfig.m_BinSize.x / MSOCConfig.m_TileSize.x,
                binHeightInTile = MSOCConfig.m_DepthBufferSize.y / MSOCConfig.m_BinSize.y / MSOCConfig.m_TileSize.y,
                binWidth = MSOCConfig.m_BinSize.x,
                binHeigh = MSOCConfig.m_BinSize.y,
                tileRanges = m_TileRanges,
                depthParams = m_DepthParams,
                invSlopes = m_InvSlope,
                vertexX = m_VertexX,
                vertexY = m_VertexY,
                isRightEdges = m_IsRightEdge,
                flatInfos = m_FlatInfo,
                tiles = m_Tiles
            };
            return binRasterizerJob.Schedule(MSOCConfig.m_BinSize.x * MSOCConfig.m_BinSize.y, 1, prepareRasterizeTriangleDataJobHandle);
        }


        private JobHandle OcclusionCulling()
        {
            var occlusionCullingJob = new OcclusionCulling
            {
                occludersSize = m_OccluderSize,
                depthBufferWidth = MSOCConfig.m_DepthBufferSize.x,
                depthBufferHeight = MSOCConfig.m_DepthBufferSize.y,
                tileHeight = MSOCConfig.m_DepthBufferSize.x / MSOCConfig.m_TileSize.x,
                vpMatrix = m_VpMatrix,
                bounds = m_CullingSystem.bounds,
                occluderFlags = m_OccluderFlags,
                tiles = m_Tiles,
                cullingResults = m_CullingSystem.cullingResults,
            };
            return occlusionCullingJob.Schedule(m_CullingSystem.bounds.Length, 64);
        }


        public void ClearTiles()
        {
            unsafe
            {
                var defaultTile = new Tile
                {
                    bitmask = uint4.zero,
                    z = 1.0f
                };
                UnsafeUtility.MemCpyReplicate(m_Tiles.GetUnsafePtr(), &defaultTile, sizeof(Tile), m_Tiles.Length);
            }
        }

        public Tile[] GetTiles()
        {
            return m_Tiles.ToArray();
        }
    }
}