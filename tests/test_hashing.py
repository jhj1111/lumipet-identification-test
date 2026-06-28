import tempfile
import os
from reid.utils.hashing import calculate_md5

def test_calculate_md5():
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(b"test cat image data")
        temp_path = f.name
        
    try:
        expected_md5 = "b41e30c559a58874c7fdcbb6578813e5"
        assert calculate_md5(temp_path) == expected_md5
    finally:
        os.remove(temp_path)
