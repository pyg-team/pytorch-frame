import os
import os.path as osp
import ssl
import sys
import urllib.request
from typing import Optional


def download_url(
    url: str,
    root: str,
    filename: Optional[str] = None,
    *,
    log: bool = True,
) -> str:
    r"""Downloads the content of :obj:`url` to the specified folder
    :obj:`root`.

    Args:
        url (str): The URL.
        root (str): The root folder.
        filename (str, optional): If set, will rename the downloaded file.
            (default: :obj:`None`)
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    if filename is None:
        filename = url.rpartition('/')[2]
        if filename[0] != '?':
            filename.split('?')[0]

    path = osp.join(root, filename)

    if osp.exists(path):
        return path

    if log and 'pytest' not in sys.modules:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(root, exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path
