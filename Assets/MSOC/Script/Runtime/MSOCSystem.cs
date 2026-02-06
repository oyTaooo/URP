#define DEBUG_MSOC
using System;
using System.Linq;
using System.Collections.Generic;
using Unity.Burst;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;


namespace MSOC
{
    public class MSOCSystem : MonoBehaviour
    {
        public Camera mainCamera { get { return m_Camera; } }
        public MeshRenderer[] occluderMeshRenders { get { return m_OccluderMeshRenders; } }
        public MeshRenderer[] occludeMeshRenders { get { return m_OccludeMeshRenders; } }
        public MaskedOcclusionCulling maskedOcclusionCulling { get { return m_MaskedOcclusionCulling; } }
        public CullingVisualizer cullingVisualizer { get { return m_CullingVisualizer; } }
        public MeshRenderer[] meshRenders { get { return m_MeshRenders; } }
        public NativeArray<bool> cullingResults { get { return m_CullingResults; } }
        public NativeArray<AABB> bounds { get { return m_Bounds; } }



        private MeshRenderer[] m_OccluderMeshRenders;

        private MeshRenderer[] m_OccludeMeshRenders;

        private MeshRenderer[] m_MeshRenders;

        private Camera m_Camera;

        private readonly Plane[] m_PlanesArray = new Plane[MSOCConfig.m_NumFrustumPlanes];

        private MaskedOcclusionCulling m_MaskedOcclusionCulling;

        //native memory
        private NativeArray<bool> m_CullingResults; //true is invisible

        private NativeArray<AABB> m_Bounds;

        private NativeArray<Plane> m_Planes;

        private NativeReference<FrustumPlanePackets> m_PlanePackets;

        //private NativeReference<OcclusionData> m_OcclusionData;

        // debug
        private CullingVisualizer m_CullingVisualizer;

        private void Awake()
        {
            m_MaskedOcclusionCulling = new MaskedOcclusionCulling(this);
#if DEBUG_MSOC
            m_CullingVisualizer = new CullingVisualizer(this);
#endif
        }

        void Start()
        {
            CollectOccluders();
                
            m_MaskedOcclusionCulling.Initialize();
                
            m_Camera = Camera.main;


            m_CullingResults = new NativeArray<bool>(m_MeshRenders.Length, Allocator.Persistent);

            m_Bounds = new NativeArray<AABB>(m_MeshRenders.Length, Allocator.Persistent);

            m_Planes = new NativeArray<Plane>(MSOCConfig.m_NumFrustumPlanes, Allocator.Persistent);

            m_PlanePackets = new NativeReference<FrustumPlanePackets>(Allocator.Persistent);

            for (int i = 0; i < m_MeshRenders.Length; i++)
            {
                m_Bounds[i] = m_MeshRenders[i].bounds.ToAABB();
            }
        }

        void LateUpdate()
        {
            FrustumCulling();

            m_MaskedOcclusionCulling.Cull();

#if DEBUG_MSOC
            m_CullingVisualizer.VisualizeCullResult();
#endif
        }

        public void OnDisable()
        {
            m_MaskedOcclusionCulling.Dispose();

            m_CullingResults.Dispose();

            m_Bounds.Dispose();

            m_Planes.Dispose();

            m_PlanePackets.Dispose();
        }

        private void FrustumCulling()
        {
            unsafe
            {
                GeometryUtility.CalculateFrustumPlanes(m_Camera, m_PlanesArray);
                fixed (Plane* pPlanesArray = m_PlanesArray)
                {
                    UnsafeUtility.MemCpy(m_Planes.GetUnsafePtr(), pPlanesArray, sizeof(Plane) * m_PlanesArray.Length);
                }
            }

            var preparePlanesJob = new PreparePlanePacketsJob
            {
                planes = m_Planes,
                planePackets = m_PlanePackets
            };
            var preparePlanesHandle = preparePlanesJob.Schedule();

            var frustumCullingJob = new FrustumCulling
            {
                planePackets = m_PlanePackets,
                bounds = m_Bounds,
                cullingResults = m_CullingResults
            };

            var frustumCullingJobHandle = frustumCullingJob.Schedule(m_Bounds.Length, 64, preparePlanesHandle);
            frustumCullingJobHandle.Complete();
        }


        private void CollectOccluders()
        {
            List<MeshRenderer> occluderList = new List<MeshRenderer>();
            HashSet<MeshRenderer> occluderSet = new HashSet<MeshRenderer>();

            // 通过 Tag 查找所有 Occluder 物体
            GameObject[] occluderObjects = GameObject.FindGameObjectsWithTag("Occluder");
            Debug.Log($"[MSOC] Found {occluderObjects.Length} Occluder Objects.");
            foreach (var obj in occluderObjects)
            {
                // 获取该物体及其子物体上的所有 MeshRenderer
                MeshRenderer[] renderers = obj.GetComponents<MeshRenderer>();
                foreach (var renderer in renderers)
                {
                    if (occluderSet.Add(renderer))
                    {
                        occluderList.Add(renderer);
                    }
                }
            }

            // Occludee: 直接获取场景中所有 MeshRenderer，排除已作为 Occluder 的
            MeshRenderer[] allRenderers = FindObjectsOfType<MeshRenderer>();
            List<MeshRenderer> occludeeList = new List<MeshRenderer>();
            foreach (var renderer in allRenderers)
            {
                if (!occluderSet.Contains(renderer))
                {
                    occludeeList.Add(renderer);
                }
            }

            m_OccluderMeshRenders = occluderList.ToArray();
            m_OccludeMeshRenders = occludeeList.ToArray();



            // 合并所有 renderer
            m_MeshRenders = new MeshRenderer[m_OccluderMeshRenders.Length + m_OccludeMeshRenders.Length];
            int idx = 0;
            foreach (var meshRenderer in m_OccluderMeshRenders)
            {
                m_MeshRenders[idx++] = meshRenderer;
            }
            foreach (var meshRenderer in m_OccludeMeshRenders)
            {
                m_MeshRenders[idx++] = meshRenderer;
            }

            Debug.Log($"[MSOC] Collected {m_OccluderMeshRenders.Length} Occluders, {m_OccludeMeshRenders.Length} Occludees, {m_MeshRenders.Length} Total Renderers.");
        }
    }
}

