import { describe, it, expect } from 'vitest';
import {
  ALL_SOURCES_VALUE,
  browseRowKey,
  buildBrowseBody,
  escapeCsv,
  isNameQueryTooShort,
  rowsToCsv,
  safeExternalUrl,
  type BrowseRow,
} from './browseTableUtils';

describe('escapeCsv', () => {
  it('leaves plain values untouched', () => {
    expect(escapeCsv('hello')).toBe('hello');
    expect(escapeCsv('')).toBe('');
    expect(escapeCsv(null)).toBe('');
    expect(escapeCsv(undefined)).toBe('');
  });

  it('quotes values with commas, quotes and newlines', () => {
    expect(escapeCsv('a,b')).toBe('"a,b"');
    expect(escapeCsv('say "hi"')).toBe('"say ""hi"""');
    expect(escapeCsv('line1\nline2')).toBe('"line1\nline2"');
  });

  it.each(['=', '+', '-', '@', '\t', '\r'])(
    'neutralizes a cell starting with %j',
    (prefix) => {
      const value = `${prefix}HYPERLINK("http://evil.example","x")`;
      const escaped = escapeCsv(value);
      // Leading single quote inside a quoted cell: the spreadsheet shows text, not a formula.
      expect(escaped.startsWith(`"'${prefix}`)).toBe(true);
      expect(escaped.endsWith('"')).toBe(true);
    },
  );

  it('neutralizes formulas and still escapes embedded quotes', () => {
    expect(escapeCsv('=1+1')).toBe(`"'=1+1"`);
    expect(escapeCsv('=CMD("x")')).toBe(`"'=CMD(""x"")"`);
  });

  it('does not touch formula characters that are not at the start', () => {
    expect(escapeCsv('a=b')).toBe('a=b');
    expect(escapeCsv('foo@bar.com')).toBe('foo@bar.com');
  });

  it('serializes non-string values as JSON', () => {
    expect(escapeCsv(42)).toBe('42');
    expect(escapeCsv({ a: 1, b: 2 })).toBe('"{""a"":1,""b"":2}"');
  });
});

describe('rowsToCsv', () => {
  it('neutralizes formulas in every column of an exported row', () => {
    const csv = rowsToCsv([
      { entity_id: '@id', name: '=evil()', web_url: 'https://example.com/a' },
    ]);
    const [header, line] = csv.split('\n');
    expect(header).toBe(
      'entity_id,name,entity_type,source_name,sync_id,created_at,updated_at,web_url',
    );
    expect(line).toBe(`"'@id","'=evil()",,,,,,https://example.com/a`);
  });
});

describe('safeExternalUrl', () => {
  it('accepts http and https URLs', () => {
    expect(safeExternalUrl('https://example.com/x?y=1')).toBe('https://example.com/x?y=1');
    expect(safeExternalUrl('http://example.com')).toBe('http://example.com/');
    expect(safeExternalUrl('  HTTPS://Example.com/a  ')).toBe('https://example.com/a');
  });

  it.each([
    'javascript:alert(1)',
    'JaVaScRiPt:alert(1)',
    ' javascript:alert(1)',
    'data:text/html,<script>alert(1)</script>',
    'vbscript:msgbox(1)',
    'file:///etc/passwd',
    'ftp://example.com/file',
    '//evil.example.com',
    '/relative/path',
    'not a url',
  ])('rejects %j', (value) => {
    expect(safeExternalUrl(value)).toBeNull();
  });

  it('returns null for empty values', () => {
    expect(safeExternalUrl(null)).toBeNull();
    expect(safeExternalUrl(undefined)).toBeNull();
    expect(safeExternalUrl('')).toBeNull();
  });
});

describe('browseRowKey', () => {
  const row = (entity_id: string, sync_id?: string, source_name?: string): BrowseRow => ({
    entity_id,
    airweave_system_metadata: { sync_id, source_name },
  });

  it('differs for the same entity_id from different syncs', () => {
    expect(browseRowKey(row('42', 'sync-a'))).not.toBe(browseRowKey(row('42', 'sync-b')));
  });

  it('is stable for the same row', () => {
    expect(browseRowKey(row('42', 'sync-a'))).toBe(browseRowKey(row('42', 'sync-a')));
  });

  it('falls back to source_name when sync_id is missing', () => {
    expect(browseRowKey(row('42', undefined, 'slack'))).not.toBe(
      browseRowKey(row('42', undefined, 'notion')),
    );
  });

  it('cannot collide through separator characters in ids', () => {
    expect(browseRowKey(row('b', 'a:'))).not.toBe(browseRowKey(row(':b', 'a')));
  });
});

describe('isNameQueryTooShort', () => {
  it('flags a single non-blank character', () => {
    expect(isNameQueryTooShort('a')).toBe(true);
    expect(isNameQueryTooShort(' a ')).toBe(true);
  });

  it('allows empty, blank and long-enough queries', () => {
    expect(isNameQueryTooShort('')).toBe(false);
    expect(isNameQueryTooShort('   ')).toBe(false);
    expect(isNameQueryTooShort('ab')).toBe(false);
  });
});

describe('buildBrowseBody', () => {
  it('sends no filters for all sources and an empty query', () => {
    expect(buildBrowseBody(50, 0, ALL_SOURCES_VALUE, '')).toEqual({ limit: 50, offset: 0 });
  });

  it('adds the sync id and the trimmed name query', () => {
    expect(buildBrowseBody(200, 400, 'sync-1', '  report ')).toEqual({
      limit: 200,
      offset: 400,
      sync_ids: ['sync-1'],
      name_query: 'report',
    });
  });

  it('omits a name query below the backend minimum', () => {
    expect(buildBrowseBody(50, 0, ALL_SOURCES_VALUE, 'a')).toEqual({ limit: 50, offset: 0 });
  });
});
