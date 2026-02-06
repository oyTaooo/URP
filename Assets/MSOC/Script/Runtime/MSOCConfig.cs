using System.Collections.Generic;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine;

namespace MSOC
{
    public static class MSOCConfig
    {
        public static readonly int m_NumFrustumPlanes = 6;
           
        public static readonly Vector2Int m_TileSize = new Vector2Int(32, 4);

        public static readonly Vector2Int mSubTileSize = new Vector2Int(8, 4);

        public static readonly Vector2Int m_DepthBufferSize = new Vector2Int(256, 160);

        public static readonly Vector2Int m_BinSize = new Vector2Int(4, 1);

        public static readonly int m_MaxRasterizeTriangleSize = 20000;

        public static readonly float m_CoverageThreshold = 0.05f;
    }
}