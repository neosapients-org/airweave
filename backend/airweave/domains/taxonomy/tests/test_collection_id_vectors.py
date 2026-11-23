"""derive_collection_id reproduces Neo's TypeScript deriveCollectionId.

The expected values were produced by running the committed packages/backend/security/src/collection-id.ts
(Neo base afab4f53) under Node, not by this module. Cases: ordinary and UUID ids, case, delimiters,
whitespace, accents, composed vs decomposed characters, CJK, characters outside the BMP (UTF-16 length),
and empty ids (the original throws).
"""

import pytest

from airweave.domains.taxonomy.artifacts import derive_collection_id

VECTORS = [
    ('t-acme', 'ws-a', 'neo-7657b3ce4523258a6b4493f959c7d38a6b08f952858f78e7'),
    ('T-ACME', 'ws-a', 'neo-71df74d59fbc2cef3d434ab22e7e06d210c39090240e38f8'),
    ('t-acme', 'WS-A', 'neo-58fb6a09d3ff01b944f2c0954ff287809f46d2489a85c533'),
    ('tenant-1', '3f2a9c1e-7b44-4d0a-9e61-0c5b8d2f4a77', 'neo-dfda6b195bfcd6b3008fb4845758e5e0666849fc1e476373'),
    ('a:b', 'c', 'neo-8a37856518dacc57691927d22e62feb58519471ad25e0fc4'),
    ('a', 'b:c', 'neo-b8d47370810d318bc6f70a5c93e34b860ba61e7638b47e07'),
    ('a:', 'b', 'neo-d48ad4031ad95ce2943c1cf3bd7ac896bc5f1564a4070795'),
    ('a', ':b', 'neo-412faa5883f193a63243c93514bb6f148d0552be1e4eacee'),
    (' t-acme', 'ws-a', 'neo-8760f8da8226bffd024d3448336d24ce2511f540f2ab3eeb'),
    ('t-acme ', 'ws-a', 'neo-6e7b9a3434a79a6b63eb7cca6873ab4fe8f4d67f68695b46'),
    ('caf\u00e9', 'ws-a', 'neo-2e6ec60a52699bb7ebf57fd7307c701500f527cf6dede1be'),
    ('t-acme', 'w\u00f6rkspace', 'neo-c07ed5328f6557cddede1ca99604b04b6ca1405fee4c9564'),
    ('e\u0301', 'ws', 'neo-d1b8115100b6f0f7c2d702c351b646ea9a0ed0b06fcb83d4'),
    ('\u00e9', 'ws', 'neo-2103835a8ff86323e4e66121e6f87b39c1247c241e1cccfd'),
    ('\u65e5\u672c', 'ws', 'neo-6807e94175f723c2ba8ed9f4e699659d172b18ecb2d0a41e'),
    ('t\U0001f600', 'ws', 'neo-62a0ad34816c82ed565029bbf4070a8525ad6f240603cbc4'),
    ('t', 'w\U0001f600\U0001f600', 'neo-450cd2112869fe73f7def6e7e6dfd56c7da13c094613ce87'),
    ('', 'ws-a', None),
    ('t-acme', '', None),
    ('', '', None),
]


@pytest.mark.parametrize("tenant,workspace,expected", VECTORS)
def test_matches_the_typescript_original(tenant, workspace, expected):
    if expected is None:
        with pytest.raises(ValueError):
            derive_collection_id(tenant, workspace)
    else:
        assert derive_collection_id(tenant, workspace) == expected


def test_composed_and_decomposed_characters_are_different_ids():
    assert derive_collection_id("\u00e9", "ws") != derive_collection_id("e\u0301", "ws")


@pytest.mark.parametrize(
    "tenant, workspace, expected",
    [
        ("t\ud800", "w", "neo-2ac8df3f0185a41df5edbadd6106c3cf966c35834ddbbdc4"),
        ("t", "w\udc00", "neo-b2e2263776ff0f4a8d6b342f00f0f3cefb401b025d49e78a"),
    ],
)
def test_an_isolated_surrogate_matches_node(tenant, workspace, expected):
    """Values produced by the TypeScript original under Node (isolated surrogate hashed as U+FFFD, length 1)."""
    assert derive_collection_id(tenant, workspace) == expected


def test_an_isolated_surrogate_is_the_same_id_as_the_replacement_character():
    assert derive_collection_id("t\ud800", "w") == derive_collection_id("t\ufffd", "w")
