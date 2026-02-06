using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace MSOC
{
    /// <summary>
    /// SIMD optimized plane packet - stores 4 planes packed for parallel processing
    /// Reference: BrgRenderSystem/FrustumPlanes.cs PlanePacket4
    /// </summary>
    public struct PlanePacket4
    {
        public float4 nx;
        public float4 ny;
        public float4 nz;
        public float4 d;
        // Pre-computed absolute values to avoid recalculating per instance
        public float4 nxAbs;
        public float4 nyAbs;
        public float4 nzAbs;

        public static PlanePacket4 Create(NativeArray<Plane> planes, int offset, int planeCount)
        {
            int limit = planeCount - 1;
            Plane p0 = planes[math.min(offset + 0, limit)];
            Plane p1 = planes[math.min(offset + 1, limit)];
            Plane p2 = planes[math.min(offset + 2, limit)];
            Plane p3 = planes[math.min(offset + 3, limit)];

            var packet = new PlanePacket4
            {
                nx = new float4(p0.normal.x, p1.normal.x, p2.normal.x, p3.normal.x),
                ny = new float4(p0.normal.y, p1.normal.y, p2.normal.y, p3.normal.y),
                nz = new float4(p0.normal.z, p1.normal.z, p2.normal.z, p3.normal.z),
                d = new float4(p0.distance, p1.distance, p2.distance, p3.distance)
            };
            packet.nxAbs = math.abs(packet.nx);
            packet.nyAbs = math.abs(packet.ny);
            packet.nzAbs = math.abs(packet.nz);

            return packet;
        }
    }

    /// <summary>
    /// Pre-computed plane packets for frustum culling (6 planes = 2 packets)
    /// </summary>
    public struct FrustumPlanePackets
    {
        public PlanePacket4 packet0; // planes 0-3
        public PlanePacket4 packet1; // planes 4-5 (duplicated to fill packet)

        public static FrustumPlanePackets Create(NativeArray<Plane> planes)
        {
            return new FrustumPlanePackets
            {
                packet0 = PlanePacket4.Create(planes, 0, planes.Length),
                packet1 = PlanePacket4.Create(planes, 4, planes.Length)
            };
        }
    }

    /// <summary>
    /// Job to pre-compute plane packets from raw planes (runs once per frame)
    /// </summary>
    [BurstCompile]
    public struct PreparePlanePacketsJob : IJob
    {
        [ReadOnly] public NativeArray<Plane> planes;
        [WriteOnly] public NativeReference<FrustumPlanePackets> planePackets;

        public void Execute()
        {
            planePackets.Value = FrustumPlanePackets.Create(planes);
        }
    }

    /// <summary>
    /// SIMD optimized frustum culling job
    /// Uses PlanePacket4 to process 4 planes in parallel using float4 SIMD operations
    /// </summary>
    [BurstCompile]
    public struct FrustumCulling : IJobParallelFor
    {
        [ReadOnly] public NativeReference<FrustumPlanePackets> planePackets;
        [ReadOnly] public NativeArray<AABB> bounds;
        [WriteOnly] public NativeArray<bool> cullingResults;

        public void Execute(int index)
        {
            AABB bound = bounds[index];

            float4 cx = bound.center.xxxx;
            float4 cy = bound.center.yyyy;
            float4 cz = bound.center.zzzz;

            float4 ex = bound.extents.xxxx;
            float4 ey = bound.extents.yyyy;
            float4 ez = bound.extents.zzzz;

            var packets = planePackets.Value;
            {
                PlanePacket4 p = packets.packet0;
                float4 distances = p.nx * cx + p.ny * cy + p.nz * cz + p.d;
                float4 radii = p.nxAbs * ex + p.nyAbs * ey + p.nzAbs * ez;
                bool4 isCulled = distances + radii < float4.zero;

                if (math.any(isCulled))
                {
                    cullingResults[index] = true;
                    return;
                }
            }
            {
                PlanePacket4 p = packets.packet1;
                float4 distances = p.nx * cx + p.ny * cy + p.nz * cz + p.d;
                float4 radii = p.nxAbs * ex + p.nyAbs * ey + p.nzAbs * ez;
                bool4 isCulled = distances + radii < float4.zero;

                if (isCulled.x || isCulled.y)
                {
                    cullingResults[index] = true;
                    return;
                }
            }

            cullingResults[index] = false;
        }
    }
}

