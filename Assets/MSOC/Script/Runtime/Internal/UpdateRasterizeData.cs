using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace MSOC
{
    [BurstCompile]
    public struct UpdateRasterizeDataJob : IJob
    {
        public int occluderSize;

        public int maxRasterizeTriangleSize;

        [ReadOnly] public NativeArray<int> occluderTriangleSize;

        [WriteOnly] public NativeArray<int> offsets;

        [WriteOnly] public NativeArray<int> rasterizeTriangleSize;

        public NativeArray<OccluderSortInfo> occluderInfos;

        public void Execute()
        {
            occluderInfos.Sort(new OccluderSortInfoComparer());
            int triangleSize = 0;
            int maxTriangleSize = maxRasterizeTriangleSize;

            for(int i = 0; i < occluderSize; i++)
            {
                OccluderSortInfo occluderInfo = occluderInfos[i];

                if (occluderInfo.coverage <= 0.0f)
                    break;

                int index = occluderInfo.index;
                offsets[index] = triangleSize;

                int size = math.min(occluderTriangleSize[index], maxTriangleSize);
                triangleSize += size;
                maxTriangleSize -= size;

                occluderInfo.rasterizeTriangleSize = size;
                occluderInfos[i] = occluderInfo;
            }

            rasterizeTriangleSize[0] = triangleSize;
        }
    }

}
