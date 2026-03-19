# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Regression tests for EXR channel decoding and segmentation postprocess.
# These tests cover the Blender 5 EEVEE object-index pipeline edge cases.

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# Mock OpenEXR module for isolated testing
class MockOpenEXR:
    class InputFile:
        def __init__(self, path):
            self.path = path
            self.header_data = {}
        
        def header(self):
            return self.header_data
    
    class ChannelInfo:
        def __init__(self, type_str="FLOAT"):
            self.type = type_str
    
    @staticmethod
    def OutputFile(path, header):
        return MagicMock()


class TestEXRChannelDecoding:
    """Test EXR loading with various channel layouts."""

    def test_load_exr_alpha_first_channel_not_data(self):
        """Test that load_exr doesn't mistake alpha for actual IndexOB data.
        
        When the first channel in an EXR is an alpha channel (IndexOB.A = 1.0),
        load_seg_mask should explicitly read IndexOB.R instead.
        """
        from infinigen.core.rendering.post_render import load_seg_mask
        
        # Create a temporary EXR-like structure
        with tempfile.TemporaryDirectory() as tmpdir:
            exr_path = Path(tmpdir) / "test_indexob.exr"
            
            # Mock OpenEXR to return alpha channel first
            mock_channels = {
                "IndexOB.A": MockOpenEXR.ChannelInfo("FLOAT"),
                "IndexOB.R": MockOpenEXR.ChannelInfo("FLOAT"),
                "IndexOB.G": MockOpenEXR.ChannelInfo("FLOAT"),
                "IndexOB.B": MockOpenEXR.ChannelInfo("FLOAT"),
            }
            
            # Simulate IndexOB.R containing actual object indices
            # IndexOB.A would be all 1.0 if we used it
            expected_index_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            
            with patch('infinigen.core.rendering.post_render.OpenEXR.InputFile') as mock_input:
                mock_instance = MagicMock()
                mock_instance.header.return_value = {
                    "channels": mock_channels,
                    "dataWindow": MagicMock(min=MagicMock(y=0, x=0), max=MagicMock(y=1, x=2))
                }
                
                def side_effect(name, dtype):
                    if name == "IndexOB.R":
                        return expected_index_data.tobytes()
                    elif name == "IndexOB.A":
                        return np.ones((2, 3), dtype=np.float32).tobytes()
                    return np.zeros((2, 3), dtype=np.float32).tobytes()
                
                mock_instance.channel.side_effect = side_effect
                mock_input.return_value = mock_instance
                
                result = load_seg_mask(exr_path)
                
                # Should have read IndexOB.R, not IndexOB.A
                np.testing.assert_array_equal(result, expected_index_data.astype(np.int64))


    def test_load_uniq_inst_rgb_channel_order(self):
        """Test that UniqueInstances RGB channels are read in correct order.
        
        Flat-shading renders emission colors to UniqueInstances.R/G/B channels.
        The channels must be read in RGB order, not arbitrary order.
        The function scales emission floats to [0, 65535] uint16.
        """
        from infinigen.core.rendering.post_render import load_uniq_inst
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exr_path = Path(tmpdir) / "test_uniq_inst.exr"
            
            # Define distinct RGB colors as normalized floats [0, 1]
            # After scaling by 65535, 1.0 -> 65535, 0.5 -> 32767, etc.
            r_channel = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
            g_channel = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
            b_channel = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
            
            mock_channels = {
                "UniqueInstances.R": MockOpenEXR.ChannelInfo("FLOAT"),
                "UniqueInstances.G": MockOpenEXR.ChannelInfo("FLOAT"),
                "UniqueInstances.B": MockOpenEXR.ChannelInfo("FLOAT"),
            }
            
            with patch('infinigen.core.rendering.post_render.OpenEXR.InputFile') as mock_input:
                mock_instance = MagicMock()
                mock_instance.header.return_value = {
                    "channels": mock_channels,
                    "dataWindow": MagicMock(min=MagicMock(y=0, x=0), max=MagicMock(y=1, x=1))
                }
                
                def side_effect(name, dtype):
                    if name == "UniqueInstances.R":
                        return r_channel.tobytes()
                    elif name == "UniqueInstances.G":
                        return g_channel.tobytes()
                    elif name == "UniqueInstances.B":
                        return b_channel.tobytes()
                    return np.zeros((2, 2), dtype=np.float32).tobytes()
                
                mock_instance.channel.side_effect = side_effect
                mock_input.return_value = mock_instance
                
                result = load_uniq_inst(exr_path)
                
                # Verify shape is (H, W, 3)
                assert result.shape == (2, 2, 3), f"Expected (2, 2, 3), got {result.shape}"
                
                # After scaling by 65535, 1.0 -> 65535
                # Pixel (0,0) should be red (R=65535, G=0, B=0)
                assert result[0, 0, 0] == 65535, f"Red channel at [0,0,0] should be 65535, got {result[0, 0, 0]}"
                assert result[0, 0, 1] == 0, f"Green channel at [0,0,1] should be 0, got {result[0, 0, 1]}"
                assert result[0, 0, 2] == 0, f"Blue channel at [0,0,2] should be 0, got {result[0, 0, 2]}"
                
                # Pixel (0,1) should be green (R=0, G=65535, B=0)
                assert result[0, 1, 0] == 0, f"Red channel at [0,1,0] should be 0, got {result[0, 1, 0]}"
                assert result[0, 1, 1] == 65535, f"Green channel at [0,1,1] should be 65535, got {result[0, 1, 1]}"
                assert result[0, 1, 2] == 0, f"Blue channel at [0,1,2] should be 0, got {result[0, 1, 2]}"


class TestSegmentationColorization:
    """Test segmentation colorization logic."""

    def test_two_labels_distinct_colors(self):
        """Test that two different labels produce strongly distinct colors.
        
        Background (label 0) should be black (0, 0, 0).
        Non-zero labels should have high-contrast colors.
        """
        from infinigen.core.rendering.post_render import colorize_int_array
        
        # Create a 4x4 segmentation with background (0) and one object label (1)
        seg_data = np.zeros((4, 4, 1), dtype=np.int64)
        seg_data[1:3, 1:3, 0] = 1  # A 2x2 square of label 1
        
        result = colorize_int_array(seg_data)
        
        # Background should be black
        assert tuple(result[0, 0]) == (0, 0, 0), "Background should be black"
        
        # Object color should NOT be black
        obj_color = result[1, 1]
        assert not (obj_color[0] == 0 and obj_color[1] == 0 and obj_color[2] == 0), \
            f"Object label should not be black, got {obj_color}"
        
        # Object color should be high contrast (vivid, saturated)
        max_channel = max(obj_color)
        min_channel = min(obj_color)
        assert max_channel > 100, f"Object color should be vivid, got {obj_color}"

    def test_multiple_labels_different_colors(self):
        """Test that multiple labels get different colors."""
        from infinigen.core.rendering.post_render import colorize_int_array
        
        # Create segmentation with 3 distinct labels
        seg_data = np.array([
            [[1], [2], [3]],
            [[0], [1], [2]],
            [[0], [0], [1]]
        ], dtype=np.int64)
        
        result = colorize_int_array(seg_data)
        
        # Get colors for each label
        color_0 = result[0, 0]  # label 0
        color_1 = result[0, 1]  # label 1
        color_2 = result[1, 1]   # label 2
        color_3 = result[0, 2]   # label 3
        
        # All non-zero labels should have distinct colors
        assert not np.array_equal(color_1, color_2), \
            f"Labels 1 and 2 should have different colors: {color_1} vs {color_2}"
        assert not np.array_equal(color_2, color_3), \
            f"Labels 2 and 3 should have different colors: {color_2} vs {color_3}"
        assert not np.array_equal(color_1, color_3), \
            f"Labels 1 and 3 should have different colors: {color_1} vs {color_3}"


class TestEXRGeneral:
    """Test general EXR loading functionality."""

    def test_load_exr_three_channel_rgb(self):
        """Test loading standard 3-channel RGB EXR."""
        from infinigen.core.rendering.post_render import load_exr
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exr_path = Path(tmpdir) / "test_rgb.exr"
            
            # Create expected RGB data
            expected = np.zeros((100, 100, 3), dtype=np.float32)
            expected[0:50, 0:50] = [1.0, 0.0, 0.0]  # Red quadrant
            expected[50:100, 50:100] = [0.0, 1.0, 0.0]  # Green quadrant
            
            with patch('infinigen.core.rendering.post_render.cv2') as mock_cv2:
                mock_cv2.imread.return_value = None  # Force OpenEXR path
                
                with patch('infinigen.core.rendering.post_render.OpenEXR.InputFile') as mock_input:
                    with patch('infinigen.core.rendering.post_render.Path') as mock_path:
                        # Make Path().exists() return True and Path().suffix return ".exr"
                        mock_path_instance = MagicMock()
                        mock_path_instance.exists.return_value = True
                        mock_path_instance.suffix = ".exr"
                        mock_path.return_value = mock_path_instance
                        
                        mock_instance = MagicMock()
                        mock_instance.header.return_value = {
                            "channels": {
                                "R": MockOpenEXR.ChannelInfo("FLOAT"),
                                "G": MockOpenEXR.ChannelInfo("FLOAT"),
                                "B": MockOpenEXR.ChannelInfo("FLOAT"),
                            },
                            "dataWindow": MagicMock(
                                min=MagicMock(y=0, x=0),
                                max=MagicMock(y=99, x=99)
                            )
                        }
                        
                        def side_effect(name, dtype):
                            h, w = 100, 100
                            if name == "R":
                                data = expected[:, :, 0]
                            elif name == "G":
                                data = expected[:, :, 1]
                            elif name == "B":
                                data = expected[:, :, 2]
                            else:
                                data = np.zeros((h, w), dtype=np.float32)
                            return data.tobytes()
                        
                        mock_instance.channel.side_effect = side_effect
                        mock_input.return_value = mock_instance
                        
                        result = load_exr(exr_path)
                        
                        np.testing.assert_array_almost_equal(result, expected)


class TestLoadSingleChannel:
    """Test single channel EXR loading."""

    def test_load_single_channel_float(self):
        """Test loading a single-channel FLOAT EXR."""
        from infinigen.core.rendering.post_render import load_single_channel
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exr_path = Path(tmpdir) / "test_single.exr"
            
            expected = np.random.rand(100, 100).astype(np.float32)
            
            with patch('infinigen.core.rendering.post_render.OpenEXR.InputFile') as mock_input:
                mock_instance = MagicMock()
                mock_instance.header.return_value = {
                    "channels": {
                        "Depth": MockOpenEXR.ChannelInfo("FLOAT"),
                    },
                    "dataWindow": MagicMock(
                        min=MagicMock(y=0, x=0),
                        max=MagicMock(y=99, x=99)
                    )
                }
                
                mock_instance.channel.side_effect = lambda name, dtype: expected.tobytes()
                mock_input.return_value = mock_instance
                
                result = load_single_channel(exr_path)
                
                np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
