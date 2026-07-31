import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
    Check,
    ChevronLeft,
    ChevronRight,
    Copy,
    Download,
    ExternalLink,
    Loader2,
    Search as SearchIcon,
    X,
} from "lucide-react";
import { apiClient } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table";
import {
    Sheet,
    SheetContent,
    SheetDescription,
    SheetHeader,
    SheetTitle,
} from "@/components/ui/sheet";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { getAppIconUrl } from "@/lib/utils/icons";
import { useTheme } from "@/lib/theme-provider";

// ===== Types mirroring backend SearchResult =====
// Kept loose on purpose: Vespa returns optional/varying fields per source.

interface Breadcrumb {
    entity_id?: string;
    name?: string;
    entity_type?: string;
}

interface SystemMetadata {
    source_name?: string;
    entity_type?: string;
    original_entity_id?: string;
    chunk_index?: number;
    sync_id?: string;
    sync_job_id?: string;
}

interface BrowseRow {
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

interface BrowseResponse {
    results: BrowseRow[];
    total: number;
    limit: number;
    offset: number;
}

interface BrowseRequestBody {
    limit: number;
    offset: number;
    sync_ids?: string[];
    name_query?: string;
}

interface BrowseTableProps {
    collectionReadableId: string;
    sourceConnections: Array<{
        id: string;
        name: string;
        short_name: string;
        sync_id?: string;
    }>;
}

const PAGE_SIZE = 50;
const SEARCH_DEBOUNCE_MS = 250;
const EXPORT_MAX_ROWS = 1000;
// Backend caps `limit` at 200 (BrowseRequest in search_v2.py).
const EXPORT_PAGE_SIZE = 200;
// Backend requires `name_query` length >= 2 to avoid full-scan triggers on a single char.
const NAME_QUERY_MIN_LENGTH = 2;

const ALL_SOURCES_VALUE = "__all__";

const formatTimestamp = (value: string | null | undefined): string => {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
};

const buildBrowseBody = (
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

// ===== CSV / JSON helpers =====

const CSV_COLUMNS: Array<{ header: string; pick: (r: BrowseRow) => unknown }> = [
    { header: "entity_id", pick: (r) => r.entity_id },
    { header: "name", pick: (r) => r.name ?? "" },
    { header: "entity_type", pick: (r) => r.airweave_system_metadata?.entity_type ?? "" },
    { header: "source_name", pick: (r) => r.airweave_system_metadata?.source_name ?? "" },
    { header: "sync_id", pick: (r) => r.airweave_system_metadata?.sync_id ?? "" },
    { header: "created_at", pick: (r) => r.created_at ?? "" },
    { header: "updated_at", pick: (r) => r.updated_at ?? "" },
    { header: "web_url", pick: (r) => r.web_url ?? "" },
];

const escapeCsv = (value: unknown): string => {
    if (value === null || value === undefined) return "";
    const s = typeof value === "string" ? value : JSON.stringify(value);
    if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
};

const rowsToCsv = (rows: BrowseRow[]): string => {
    const head = CSV_COLUMNS.map((c) => c.header).join(",");
    const body = rows
        .map((r) => CSV_COLUMNS.map((c) => escapeCsv(c.pick(r))).join(","))
        .join("\n");
    return `${head}\n${body}\n`;
};

const triggerDownload = (filename: string, mime: string, content: string) => {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
};

// ===== Component =====

export function BrowseTable({ collectionReadableId, sourceConnections }: BrowseTableProps) {
    const { resolvedTheme } = useTheme();
    const [rows, setRows] = useState<BrowseRow[]>([]);
    const [total, setTotal] = useState(0);
    const [offset, setOffset] = useState(0);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedSyncId, setSelectedSyncId] = useState<string>(ALL_SOURCES_VALUE);
    const [activeRow, setActiveRow] = useState<BrowseRow | null>(null);

    // Reactive search bar
    const [searchInput, setSearchInput] = useState("");
    const [debouncedSearch, setDebouncedSearch] = useState("");

    // Export state
    const [isExporting, setIsExporting] = useState(false);

    // Reset offset when filters change
    useEffect(() => {
        setOffset(0);
    }, [selectedSyncId, debouncedSearch]);

    // Debounce search input
    useEffect(() => {
        const t = setTimeout(() => setDebouncedSearch(searchInput), SEARCH_DEBOUNCE_MS);
        return () => clearTimeout(t);
    }, [searchInput]);

    // Fetch page (abortable)
    useEffect(() => {
        const controller = new AbortController();
        const run = async () => {
            setIsLoading(true);
            setError(null);
            try {
                const body = buildBrowseBody(PAGE_SIZE, offset, selectedSyncId, debouncedSearch);
                const response = await apiClient.post(
                    `/collections/${collectionReadableId}/search/browse`,
                    body,
                    { signal: controller.signal },
                );
                if (!response.ok) {
                    const text = await response.text();
                    throw new Error(text || `Browse failed (${response.status})`);
                }
                const data: BrowseResponse = await response.json();
                if (!controller.signal.aborted) {
                    setRows(data.results);
                    setTotal(data.total);
                }
            } catch (e) {
                if (controller.signal.aborted) return;
                if (e instanceof Error && e.name === "AbortError") return;
                setError(e instanceof Error ? e.message : "Failed to load");
            } finally {
                if (!controller.signal.aborted) setIsLoading(false);
            }
        };
        void run();
        return () => controller.abort();
    }, [collectionReadableId, offset, selectedSyncId, debouncedSearch]);

    // sync_id → SourceConnection lookup
    const syncToConnection = useMemo(() => {
        const map = new Map<string, BrowseTableProps["sourceConnections"][number]>();
        for (const conn of sourceConnections) {
            if (conn.sync_id) map.set(conn.sync_id, conn);
        }
        return map;
    }, [sourceConnections]);

    const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    const currentPage = Math.floor(offset / PAGE_SIZE) + 1;
    const canPrev = offset > 0;
    const canNext = offset + PAGE_SIZE < total;

    const filteringApplied =
        selectedSyncId !== ALL_SOURCES_VALUE ||
        debouncedSearch.trim().length >= NAME_QUERY_MIN_LENGTH;

    const searchTrimmedLength = searchInput.trim().length;
    const searchTooShort =
        searchTrimmedLength > 0 && searchTrimmedLength < NAME_QUERY_MIN_LENGTH;

    // ===== Export =====

    const fetchAllForExport = useCallback(async (): Promise<BrowseRow[]> => {
        // Backend caps a single page at 200, so loop in chunks until we hit the export cap.
        const cap = Math.min(total, EXPORT_MAX_ROWS);
        const accumulated: BrowseRow[] = [];
        let pageOffset = 0;
        while (accumulated.length < cap) {
            const remaining = cap - accumulated.length;
            const pageLimit = Math.min(EXPORT_PAGE_SIZE, remaining);
            const body = buildBrowseBody(
                pageLimit,
                pageOffset,
                selectedSyncId,
                debouncedSearch,
            );
            const response = await apiClient.post(
                `/collections/${collectionReadableId}/search/browse`,
                body,
            );
            if (!response.ok) {
                const text = await response.text();
                throw new Error(text || `Export failed (${response.status})`);
            }
            const data: BrowseResponse = await response.json();
            if (data.results.length === 0) break;
            accumulated.push(...data.results);
            pageOffset += data.results.length;
            if (data.results.length < pageLimit) break;
        }
        return accumulated;
    }, [collectionReadableId, selectedSyncId, debouncedSearch, total]);

    const runExport = useCallback(
        async (scope: "page" | "all", format: "csv" | "json") => {
            setIsExporting(true);
            try {
                const data = scope === "page" ? rows : await fetchAllForExport();
                const stamp = new Date().toISOString().replace(/[:.]/g, "-");
                const base = `${collectionReadableId}-browse-${stamp}`;
                if (format === "csv") {
                    triggerDownload(`${base}.csv`, "text/csv;charset=utf-8", rowsToCsv(data));
                } else {
                    triggerDownload(
                        `${base}.json`,
                        "application/json;charset=utf-8",
                        JSON.stringify(data, null, 2),
                    );
                }
                if (scope === "all" && total > EXPORT_MAX_ROWS) {
                    toast.warning(
                        `Exported first ${EXPORT_MAX_ROWS.toLocaleString()} of ${total.toLocaleString()} rows. Narrow the filter to capture more.`,
                    );
                } else {
                    toast.success(`Exported ${data.length.toLocaleString()} rows`);
                }
            } catch (e) {
                toast.error(e instanceof Error ? e.message : "Export failed");
            } finally {
                setIsExporting(false);
            }
        },
        [collectionReadableId, rows, fetchAllForExport, total],
    );

    return (
        <div className="w-full">
            {/* Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
                <div className="flex items-center gap-2 min-w-0 flex-1">
                    <SearchInput
                        value={searchInput}
                        onChange={setSearchInput}
                        isPending={searchInput !== debouncedSearch || (isLoading && !!debouncedSearch)}
                        hint={searchTooShort ? "Type at least 2 characters" : null}
                    />
                    <Select value={selectedSyncId} onValueChange={setSelectedSyncId}>
                        <SelectTrigger className="h-9 w-[200px] shrink-0">
                            <SelectValue placeholder="All sources" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value={ALL_SOURCES_VALUE}>All sources</SelectItem>
                            {sourceConnections
                                .filter((c) => !!c.sync_id)
                                .map((c) => (
                                    <SelectItem key={c.id} value={c.sync_id as string}>
                                        {c.name}
                                    </SelectItem>
                                ))}
                        </SelectContent>
                    </Select>
                </div>

                <div className="flex items-center gap-2">
                    <div className="text-xs text-muted-foreground tabular-nums">
                        {total > 0 ? (
                            <>
                                {(offset + 1).toLocaleString()}–
                                {Math.min(offset + PAGE_SIZE, total).toLocaleString()} of{" "}
                                {total.toLocaleString()}
                            </>
                        ) : isLoading ? (
                            <span>Loading…</span>
                        ) : (
                            <span>0 results</span>
                        )}
                    </div>
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button
                                variant="outline"
                                size="sm"
                                className="h-9"
                                disabled={isExporting || total === 0}
                            >
                                {isExporting ? (
                                    <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                                ) : (
                                    <Download className="h-3.5 w-3.5 mr-1.5" />
                                )}
                                Export
                            </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="w-56">
                            <DropdownMenuLabel className="text-xs text-muted-foreground">
                                Current page ({rows.length.toLocaleString()})
                            </DropdownMenuLabel>
                            <DropdownMenuItem onClick={() => runExport("page", "csv")}>
                                Export page as CSV
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => runExport("page", "json")}>
                                Export page as JSON
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuLabel className="text-xs text-muted-foreground">
                                {filteringApplied ? "All matching" : "Entire collection"} (max{" "}
                                {EXPORT_MAX_ROWS.toLocaleString()})
                            </DropdownMenuLabel>
                            <DropdownMenuItem onClick={() => runExport("all", "csv")}>
                                Export all as CSV
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => runExport("all", "json")}>
                                Export all as JSON
                            </DropdownMenuItem>
                        </DropdownMenuContent>
                    </DropdownMenu>
                </div>
            </div>

            {/* Table */}
            <div className="rounded-md border bg-card">
                <Table>
                    <TableHeader>
                        <TableRow className="hover:bg-transparent">
                            <TableHead className="w-[44%]">Name</TableHead>
                            <TableHead className="w-[18%]">Type</TableHead>
                            <TableHead className="w-[18%]">Source</TableHead>
                            <TableHead className="w-[16%]">Updated</TableHead>
                            <TableHead className="w-[4%] text-right" />
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {isLoading && rows.length === 0 ? (
                            Array.from({ length: 8 }).map((_, i) => (
                                <TableRow key={`skeleton-${i}`}>
                                    {Array.from({ length: 5 }).map((__, j) => (
                                        <TableCell key={j}>
                                            <Skeleton className="h-4 w-full" />
                                        </TableCell>
                                    ))}
                                </TableRow>
                            ))
                        ) : rows.length === 0 ? (
                            <TableRow className="hover:bg-transparent">
                                <TableCell
                                    colSpan={5}
                                    className="h-32 text-center text-sm text-muted-foreground"
                                >
                                    {error
                                        ? `Error: ${error}`
                                        : filteringApplied
                                          ? "No entities match these filters."
                                          : "No entities to browse yet."}
                                </TableCell>
                            </TableRow>
                        ) : (
                            rows.map((row) => {
                                const meta = row.airweave_system_metadata ?? {};
                                const conn = meta.sync_id
                                    ? syncToConnection.get(meta.sync_id)
                                    : undefined;
                                const sourceLabel = conn?.name ?? meta.source_name ?? "—";
                                const shortName = conn?.short_name ?? meta.source_name;
                                return (
                                    <TableRow
                                        key={row.entity_id}
                                        className="cursor-pointer group"
                                        onClick={() => setActiveRow(row)}
                                    >
                                        <TableCell className="font-medium align-top">
                                            <div
                                                className="truncate max-w-[520px]"
                                                title={row.name ?? row.entity_id}
                                            >
                                                {row.name || row.entity_id}
                                            </div>
                                            {row.textual_representation && (
                                                <div className="text-xs text-muted-foreground truncate max-w-[520px] mt-0.5">
                                                    {row.textual_representation}
                                                </div>
                                            )}
                                        </TableCell>
                                        <TableCell className="align-top">
                                            <Badge
                                                variant="secondary"
                                                className="font-mono text-[10px] font-normal"
                                            >
                                                {meta.entity_type || "—"}
                                            </Badge>
                                        </TableCell>
                                        <TableCell className="align-top">
                                            <div className="flex items-center gap-2">
                                                {shortName && (
                                                    <img
                                                        src={getAppIconUrl(shortName, resolvedTheme)}
                                                        alt={sourceLabel}
                                                        className="h-4 w-4 object-contain shrink-0"
                                                    />
                                                )}
                                                <span className="text-sm truncate">{sourceLabel}</span>
                                            </div>
                                        </TableCell>
                                        <TableCell className="text-sm text-muted-foreground tabular-nums align-top">
                                            {formatTimestamp(row.updated_at ?? row.created_at)}
                                        </TableCell>
                                        <TableCell className="text-right align-top">
                                            {row.web_url ? (
                                                <a
                                                    href={row.web_url}
                                                    target="_blank"
                                                    rel="noreferrer noopener"
                                                    onClick={(e) => e.stopPropagation()}
                                                    className={cn(
                                                        "inline-flex h-7 w-7 items-center justify-center rounded",
                                                        "text-muted-foreground/60 group-hover:text-muted-foreground",
                                                        "hover:!text-foreground hover:bg-muted",
                                                    )}
                                                    title="Open in source"
                                                >
                                                    <ExternalLink className="h-3.5 w-3.5" />
                                                </a>
                                            ) : (
                                                <span className="text-xs text-muted-foreground/40">
                                                    —
                                                </span>
                                            )}
                                        </TableCell>
                                    </TableRow>
                                );
                            })
                        )}
                    </TableBody>
                </Table>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between mt-3">
                <div className="text-xs text-muted-foreground tabular-nums">
                    Page {currentPage.toLocaleString()} of {totalPages.toLocaleString()}
                </div>
                <div className="flex items-center gap-2">
                    <Button
                        variant="outline"
                        size="sm"
                        disabled={!canPrev || isLoading}
                        onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
                    >
                        <ChevronLeft className="h-4 w-4" />
                        Prev
                    </Button>
                    <Button
                        variant="outline"
                        size="sm"
                        disabled={!canNext || isLoading}
                        onClick={() => setOffset((o) => o + PAGE_SIZE)}
                    >
                        Next
                        <ChevronRight className="h-4 w-4" />
                    </Button>
                </div>
            </div>

            <RowDrawer row={activeRow} onClose={() => setActiveRow(null)} />
        </div>
    );
}

// ===== Subcomponents =====

function SearchInput({
    value,
    onChange,
    isPending,
    hint,
}: {
    value: string;
    onChange: (v: string) => void;
    isPending: boolean;
    hint?: string | null;
}) {
    const ref = useRef<HTMLInputElement>(null);
    return (
        <div className="relative flex-1 min-w-[180px] max-w-[420px]">
            <SearchIcon className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
            <Input
                ref={ref}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder="Search by name…"
                className="h-9 pl-8 pr-8"
            />
            {isPending && value && (
                <Loader2 className="absolute right-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 animate-spin text-muted-foreground" />
            )}
            {!isPending && value && (
                <button
                    type="button"
                    onClick={() => {
                        onChange("");
                        ref.current?.focus();
                    }}
                    className="absolute right-2 top-1/2 -translate-y-1/2 h-5 w-5 inline-flex items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-muted"
                    aria-label="Clear search"
                >
                    <X className="h-3 w-3" />
                </button>
            )}
            {hint && (
                <div className="absolute left-0 top-full mt-1 text-[11px] text-muted-foreground">
                    {hint}
                </div>
            )}
        </div>
    );
}

function RowDrawer({ row, onClose }: { row: BrowseRow | null; onClose: () => void }) {
    const [copied, setCopied] = useState(false);
    useEffect(() => {
        if (!row) setCopied(false);
    }, [row]);

    const onCopy = useCallback((value: string) => {
        navigator.clipboard.writeText(value).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1200);
        });
    }, []);

    return (
        <Sheet open={!!row} onOpenChange={(open) => !open && onClose()}>
            <SheetContent className="w-[640px] sm:max-w-[640px] p-0 overflow-hidden flex flex-col">
                {row && (
                    <>
                        <SheetHeader className="px-6 pt-6 pb-4 border-b sticky top-0 bg-background z-10">
                            <SheetTitle className="break-words text-base leading-snug">
                                {row.name || row.entity_id}
                            </SheetTitle>
                            <SheetDescription className="flex items-center gap-2">
                                <code className="font-mono text-[11px] break-all text-muted-foreground/80 leading-tight">
                                    {row.entity_id}
                                </code>
                                <button
                                    type="button"
                                    onClick={() => onCopy(row.entity_id)}
                                    className="shrink-0 inline-flex h-5 w-5 items-center justify-center rounded text-muted-foreground/60 hover:text-foreground hover:bg-muted"
                                    aria-label="Copy entity ID"
                                >
                                    {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                                </button>
                            </SheetDescription>
                        </SheetHeader>
                        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5 text-sm">
                            <div className="grid grid-cols-[80px_1fr] gap-y-2 gap-x-4 text-xs">
                                <DetailRow
                                    label="Type"
                                    value={row.airweave_system_metadata?.entity_type}
                                    mono
                                />
                                <DetailRow
                                    label="Source"
                                    value={row.airweave_system_metadata?.source_name}
                                />
                                <DetailRow label="Created" value={formatTimestamp(row.created_at)} />
                                <DetailRow label="Updated" value={formatTimestamp(row.updated_at)} />
                                {row.web_url && (
                                    <>
                                        <div className="text-muted-foreground">Link</div>
                                        <a
                                            href={row.web_url}
                                            target="_blank"
                                            rel="noreferrer noopener"
                                            className="text-primary hover:underline truncate"
                                        >
                                            {row.web_url}
                                        </a>
                                    </>
                                )}
                            </div>

                            {row.breadcrumbs && row.breadcrumbs.length > 0 && (
                                <Section label="Breadcrumbs">
                                    <div className="text-xs flex flex-wrap items-center gap-1">
                                        {row.breadcrumbs.map((b, i) => (
                                            <span key={i} className="flex items-center gap-1">
                                                {i > 0 && (
                                                    <span className="text-muted-foreground/50">/</span>
                                                )}
                                                <span className="text-foreground">
                                                    {b.name || b.entity_id}
                                                </span>
                                            </span>
                                        ))}
                                    </div>
                                </Section>
                            )}

                            {row.textual_representation && (
                                <Section label="Content">
                                    <pre className="whitespace-pre-wrap break-words text-xs bg-muted/50 rounded p-3 max-h-[40vh] overflow-auto leading-relaxed">
                                        {row.textual_representation}
                                    </pre>
                                </Section>
                            )}

                            {row.raw_source_fields &&
                                Object.keys(row.raw_source_fields).length > 0 && (
                                    <Section label="Raw fields">
                                        <pre className="whitespace-pre-wrap break-words text-xs bg-muted/50 rounded p-3 max-h-[30vh] overflow-auto leading-relaxed">
                                            {JSON.stringify(row.raw_source_fields, null, 2)}
                                        </pre>
                                    </Section>
                                )}
                        </div>
                    </>
                )}
            </SheetContent>
        </Sheet>
    );
}

function DetailRow({
    label,
    value,
    mono,
}: {
    label: string;
    value?: string | null;
    mono?: boolean;
}) {
    if (!value) return null;
    return (
        <>
            <div className="text-muted-foreground">{label}</div>
            <div className={cn("break-all", mono && "font-mono text-[11px]")}>{value}</div>
        </>
    );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div>
            <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground mb-2">
                {label}
            </div>
            {children}
        </div>
    );
}
