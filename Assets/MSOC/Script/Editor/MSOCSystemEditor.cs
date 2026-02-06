#if UNITY_EDITOR
using System;
using System.Diagnostics;
using UnityEditor;
using UnityEngine;


namespace MSOC
{
    [CustomEditor(typeof(MSOCSystem))]
    public class MSOCSystemEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();
            var MSOCSystem = (MSOCSystem)target;
            if (GUILayout.Button("Visualize"))
            {
                try
                {
                    MSOCSystem.cullingVisualizer.VisualizeDepthBuffer();
                }
                catch (Exception)
                {
                    UnityEngine.Debug.LogWarning("Pause the game first when async rasterize is on!");
                }
            }
        }
    }
}

#endif

