using System.IO;
using UnityEngine;


namespace MSOC
{
    public class CullingVisualizer
    {
        private MSOCSystem m_CullingSystem;

        private Texture2D depthBuffer;

        private Vector2 depthRange;

        private bool showBitmask;

        private bool reverse;

        private bool remapMinMax;

        private Color z0Color = Color.black;
        private Color z1Color = Color.white;

        public CullingVisualizer(MSOCSystem cullingSystem)
        {
            m_CullingSystem = cullingSystem;
            showBitmask = true;
        }

        public void VisualizeCullResult()
        {
            var meshRenders = m_CullingSystem.meshRenders;
            var cullingResults = m_CullingSystem.cullingResults;

            for (int i = 0; i < meshRenders.Length; i++)
            {
                var meshRenderer = meshRenders[i];
                if (cullingResults[i])
                {
                    //meshRenderer.enabled = false;
                    meshRenderer.material.color = Color.red;
                }
                else
                {
                    //meshRenderer.enabled = true;
                    meshRenderer.material.color = Color.green;
                }
            }
        }


        public void VisualizeDepthBuffer()
        {
            CreateDepthBufferIfNeeded();
            UpdateDepthBuffer();
            SaveTextureAsPNG("Assets/msoc_depth.png");
        }

        private void CreateDepthBufferIfNeeded()
        {
            if (!depthBuffer || depthBuffer.width != MSOCConfig.m_DepthBufferSize.x ||
                depthBuffer.height != MSOCConfig.m_DepthBufferSize.y)
            {
                depthBuffer = new Texture2D(MSOCConfig.m_DepthBufferSize.x, MSOCConfig.m_DepthBufferSize.y, TextureFormat.ARGB32, false)
                {
                    wrapMode = TextureWrapMode.Clamp,
                    filterMode = FilterMode.Point
                };
            }
        }

        private void SaveTextureAsPNG(string path)
        {
            var pngBytes = depthBuffer.EncodeToPNG();
            File.WriteAllBytes(path, pngBytes);
            Debug.Log("Texture saved as PNG to: " + path);
        }

        private void UpdateDepthBuffer()
        {
            var tiles = m_CullingSystem.maskedOcclusionCulling.GetTiles();
            UpdateDepthRange(tiles);
            for (var i = 0; i < tiles.Length; i++)
            {
                int widthInTiles = MSOCConfig.m_DepthBufferSize.x / MSOCConfig.m_TileSize.x;
                var tileRow = i / widthInTiles;
                var tileCol = i % widthInTiles;
                UpdateTile(tileRow, tileCol, tiles[i]);
            }
            //if (visTileBorder) ApplyTileBorder();
        }

        private void UpdateTile(int tileRow, int tileCol, in Tile tile)
        {
            // NumRowsSubTile = 1
            for (var subTileCol = 0; subTileCol < MSOCConfig.m_TileSize.x/MSOCConfig.mSubTileSize.x; subTileCol++)
            {
                var pixelRowStart = tileRow * MSOCConfig.m_TileSize.y;
                var pixelColStart = tileCol * MSOCConfig.m_TileSize.x + subTileCol * MSOCConfig.mSubTileSize.x;
                var bitmask = tile.bitmask[subTileCol];
                var z = tile.z[subTileCol];
                UpdateSubTile(pixelRowStart, pixelColStart, bitmask, z);
            }
        }

        private void UpdateSubTile(int pixelRowStart, int pixelColStart, uint bitmask, float z)
        {
            for (var row = 0; row < MSOCConfig.mSubTileSize.y; row++)
            {
                for (var col = 0; col < MSOCConfig.mSubTileSize.x; col++)
                {
                    var idx = (MSOCConfig.mSubTileSize.y - 1 - row) * MSOCConfig.mSubTileSize.x + col;
                    var bitValue = (bitmask >> (31 - idx)) & 1;
                    var pixelRow = pixelRowStart + row;
                    var pixelCol = pixelColStart + col;

                    if (showBitmask)
                    {
                        depthBuffer.SetPixel(pixelCol, pixelRow, bitValue == 1 ? z1Color : z0Color);
                        continue;
                    }

                    if (bitValue == 1)
                    {
                        z = Mathf.Clamp(z, 0.0f, 1.0f);
                        if (reverse) z = 1.0f - z;
                        if (remapMinMax) z = (z - depthRange.x) / (depthRange.y - depthRange.x);
                        depthBuffer.SetPixel(pixelCol, pixelRow, new Color(z, z, z, 1.0f));
                    }
                    else
                    {
                        depthBuffer.SetPixel(pixelCol, pixelRow, z0Color);
                    }
                }
            }
        }

        private void UpdateDepthRange(Tile[] tiles)
        {
            var minDepth = float.MaxValue;
            var maxDepth = float.MinValue;
            foreach (var tile in tiles)
            {
                for (var i = 0; i < 4; i++)
                {
                    if (Mathf.Approximately(tile.z[i], 1.0f)) continue;
                    minDepth = Mathf.Min(minDepth, tile.z[i]);
                    maxDepth = Mathf.Max(maxDepth, tile.z[i]);
                }
            }
            depthRange = new Vector2(minDepth, maxDepth);
        }
    }
}
