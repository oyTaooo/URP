using System;
using System.Collections.Generic;
using Unity.Mathematics;


namespace MSOC
{
    public struct OccluderSortInfo
    {
        public int index;
        public float coverage;
        public int rasterizeTriangleSize;
    }

    public struct OccluderSortInfoComparer : IComparer<OccluderSortInfo>
    {
        public int Compare(OccluderSortInfo x, OccluderSortInfo y)
        {
            return y.coverage.CompareTo(x.coverage); 
        }
    }
}
