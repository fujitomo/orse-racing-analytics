import pytest
from unittest.mock import patch

from horse_racing.data.process_race_data import process_race_data

def test_process_race_data_calls_processors():
    """Test that process_race_data calls both BAC and SED processors."""
    with patch('horse_racing.data.process_race_data.process_all_bac_files') as mock_bac_processor:
        with patch('horse_racing.data.process_race_data.process_all_sed_files') as mock_sed_processor:
            
            process_race_data()
            
            mock_bac_processor.assert_called_once()
            mock_sed_processor.assert_called_once()