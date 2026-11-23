"""Resolve a model repository or a pinned asset to a local cache path."""
from pathlib import Path
import re

from .errors import InputError

ID = re.compile(r'[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*')


LINGBOT_ROBOTWIN = 'robbyant/lingbot-vla-v2-6b-robotwin'
QWEN_PROCESSOR = 'hf://Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17'
PROCESSOR_FILES = ['*.json', '*.txt', '*.model', '*.jinja', '*.tiktoken']


def fetch(spec, *, revision=None, offline=False, patterns=None):
    address = spec.removeprefix('hf://')
    parts = address.split('/')
    if len(parts) < 2:
        raise InputError('Use hf://owner/repo@revision[/path]')
    repo, _, embedded = '/'.join(parts[:2]).partition('@')
    if not ID.fullmatch(repo) or ('@' in parts[1] and not embedded) or (embedded and revision and embedded != revision):
        raise InputError('Invalid Hub ID or conflicting revisions')
    subpath = parts[2:]
    if any(p in ('', '.', '..') or '\\' in p for p in subpath):
        raise InputError('Invalid relative Hub asset path')
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise InputError('Hub inputs require pip install "vvla[hub]"') from error
    try:
        subset = ['/'.join(subpath), '/'.join(subpath) + '/**'] if subpath else patterns
        root = Path(snapshot_download(repo_id=repo, revision=revision or embedded or None,
                                     local_files_only=offline, allow_patterns=subset))
    except Exception as error:
        raise InputError(f'Cannot resolve Hub repository {repo}: {error}') from error
    target = root.joinpath(*subpath)
    if not target.exists():
        raise InputError(f'Missing Hub asset {spec}')
    # Keep the snapshot path (individual files can be symlinks into blobs/).
    return str(target.absolute())


def pull(spec, *, revision=None, offline=False):
    result = fetch(spec, revision=revision, offline=offline)
    if spec.removeprefix('hf://').partition('@')[0] == LINGBOT_ROBOTWIN:
        fetch(QWEN_PROCESSOR, offline=offline, patterns=PROCESSOR_FILES)
    return result


def checkpoint_path(value, *, revision=None, offline=False):
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    if value.startswith('hf://') or ID.fullmatch(value):
        return fetch(value, revision=revision, offline=offline)
    if revision:
        raise InputError('A Hub revision requires a model repository ID')
    return str(path.absolute())
