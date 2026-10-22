// Pure helpers for BrowseTable: request body building, row keys, URL safety, CSV export.
// Kept free of React and `@/` imports so they can be unit-tested in isolation.

// ===== Types mirroring backend SearchResult =====
// Kept loose on purpose: Vespa returns optional/varying fields per source.

export interface Breadcrumb {
    entity_id?: string;
    name?: string;
    entity_type?: string;
}

export interface SystemMetadata {
    source_name?: string;
    entity_type?: string;
    original_entity_id?: string;
    chunk_index?: number;
    sync_id?: string;
    sync_job_id?: string;
}

export interface BrowseRow {
    entity_id: string;
    name?: string | null;
    textual_representation?: string | null;
    breadcrumbs?: Breadcrumb[] | null;
    created_at?: string | null;
    updated_at?: string | null;
    airweave_system_metadata?: SystemMetadata;
    web_url?: string | null;
    url?: string | null;
    raw_source_fields?: Record<string, unknown> | null;
}

export interface BrowseResponse {
    results: BrowseRow[];
    total: number;
    limit: number;
    offset: number;
}

export interface BrowseRequestBody {
    limit: number;
    offset: number;
    sync_ids?: string[];
    name_query?: string;
}

// Backend requires `name_query` length >= 2 to avoid full-scan triggers on a single char.
export const NAME_QUERY_MIN_LENGTH = 2;

export const ALL_SOURCES_VALUE = "__all__";

// ===== Name query =====

/** True when the user typed something, but too little for the backend to filter on. */
export const isNameQueryTooShort = (nameQuery: string): boolean => {
    const length = nameQuery.trim().length;
    return length > 0 && length < NAME_QUERY_MIN_LENGTH;
};

// ===== Request body =====

/**
 * Build the browse request body. Callers must not fetch while the name query is too short
 * (see isNameQueryTooShort): a short query is dropped here, which would mean "no filter".
 */
export const buildBrowseBody = (
    limit: number,
    offset: number,
    selectedSyncId: string,
    nameQuery: string,
): BrowseRequestBody => {
    const body: BrowseRequestBody = { limit, offset };
    if (selectedSyncId !== ALL_SOURCES_VALUE) {
        body.sync_ids = [selectedSyncId];
    }
    const trimmed = nameQuery.trim();
    if (trimmed.length >= NAME_QUERY_MIN_LENGTH) {
        body.name_query = trimmed;
    }
    return body;
};

// ===== Row identity =====

/**
 * Stable React key for a row. `entity_id` is only unique within one source, so the key is
 * scoped by sync id (falling back to the source name when the sync id is missing).
 */
export const browseRowKey = (row: BrowseRow): string => {
    const meta = row.airweave_system_metadata;
    const scope = meta?.sync_id ?? meta?.source_name ?? "";
    return JSON.stringify([scope, row.entity_id]);
};

// ===== External links =====

const SAFE_URL_PROTOCOLS = new Set(["http:", "https:"]);

/**
 * Return the URL if it is an absolute http(s) URL, else null. Source-controlled values such as
 * `javascript:` or `data:` URLs must never become clickable links.
 */
export const safeExternalUrl = (value: string | null | undefined): string | null => {
    if (!value) return null;
    let parsed: URL;
    try {
        parsed = new URL(value.trim());
    } catch {
        return null;
    }
    return SAFE_URL_PROTOCOLS.has(parsed.protocol) ? parsed.href : null;
};

// ===== CSV =====

export const CSV_COLUMNS: Array<{ header: string; pick: (r: BrowseRow) => unknown }> = [
    { header: "entity_id", pick: (r) => r.entity_id },
    { header: "name", pick: (r) => r.name ?? "" },
    { header: "entity_type", pick: (r) => r.airweave_system_metadata?.entity_type ?? "" },
    { header: "source_name", pick: (r) => r.airweave_system_metadata?.source_name ?? "" },
    { header: "sync_id", pick: (r) => r.airweave_system_metadata?.sync_id ?? "" },
    { header: "created_at", pick: (r) => r.created_at ?? "" },
    { header: "updated_at", pick: (r) => r.updated_at ?? "" },
    { header: "web_url", pick: (r) => r.web_url ?? "" },
];

// Spreadsheet apps evaluate cells starting with these as formulas (OWASP CSV injection).
const CSV_FORMULA_TRIGGER = /^[=+\-@\t\r]/;
const CSV_NEEDS_QUOTING = /[",\n\r]/;

/**
 * Escape one CSV cell. Values that a spreadsheet would read as a formula get a leading
 * single quote and are always quoted; values with separators or quotes are quoted.
 */
export const escapeCsv = (value: unknown): string => {
    if (value === null || value === undefined) return "";
    const raw = typeof value === "string" ? value : JSON.stringify(value);
    const isFormula = CSV_FORMULA_TRIGGER.test(raw);
    const s = isFormula ? `'${raw}` : raw;
    if (isFormula || CSV_NEEDS_QUOTING.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
};

export const rowsToCsv = (rows: BrowseRow[]): string => {
    const head = CSV_COLUMNS.map((c) => c.header).join(",");
    const body = rows
        .map((r) => CSV_COLUMNS.map((c) => escapeCsv(c.pick(r))).join(","))
        .join("\n");
    return `${head}\n${body}\n`;
};
