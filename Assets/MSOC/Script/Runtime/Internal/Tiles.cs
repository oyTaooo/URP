using System;
using Unity.Collections;
using Unity.Mathematics;


namespace MSOC
{
    // contain 1x4 SubTiles
    [Serializable]
    public struct Tile 
    {
        public float4 z;
        public uint4 bitmask;
    }

    public struct OcclusionData
    {
        public Tile[] tiles;

        public static OcclusionData Create(int tileSize)
        {
            return new OcclusionData
            {
                tiles  = new Tile[tileSize]
            };
        }
}
}