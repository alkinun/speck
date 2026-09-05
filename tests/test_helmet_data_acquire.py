import os

import pytest

from scripts.helmet_data_acquire import qualify_volume


def test_qualify_volume_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="real existing directory"):
        qualify_volume(link, 1)


def test_qualify_volume_rejects_wrong_filesystem(tmp_path):
    path = tmp_path / "open"
    path.mkdir(mode=0o755)
    os.chmod(path, 0o755)

    with pytest.raises(ValueError, match="identity"):
        qualify_volume(path, 1)
