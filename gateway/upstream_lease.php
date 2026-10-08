<?php
// Heartbeat lease for the gateway proxy. deploy/serve_colab.py re-POSTs set_upstream.php every 30 s and
// set_upstream.php stamps "updated_at" (unix s) on every POST. A state older than the TTL (default 90 s, or
// SIMPLETI_LEASE_TTL_S when it is a positive integer) means the Colab is gone: the proxy answers 503 + Retry-After
// instead of calling a dead tunnel. A missing or unreadable state is also "down".
// The state file comes only from SIMPLETI_UPSTREAM_FILE (absolute, no default, never the system temp dir):
// without it nothing is live, so the proxy fails closed.
const SIMPLETI_LEASE_TTL_DEFAULT_S = 90;
const SIMPLETI_RETRY_AFTER_S = 30;

function simpleti_state_file(): ?string
{
    $file = getenv('SIMPLETI_UPSTREAM_FILE') ?: '';
    return strncmp($file, '/', 1) === 0 ? $file : null;
}

function simpleti_lease_ttl(): int
{
    $raw = getenv('SIMPLETI_LEASE_TTL_S');
    return is_string($raw) && preg_match('/^[1-9][0-9]{0,5}$/D', $raw) === 1 ? (int) $raw : SIMPLETI_LEASE_TTL_DEFAULT_S;
}

function simpleti_read_state(string $file): ?array
{
    $state = is_file($file) ? json_decode((string) @file_get_contents($file), true) : null;
    return is_array($state) ? $state : null;
}

function simpleti_live_upstream(?array $state, int $now): ?string
{
    $url = $state['upstream_url'] ?? null;
    $at = $state['updated_at'] ?? null;
    if (!is_string($url) || $url === '' || !is_int($at) || $now - $at > simpleti_lease_ttl()) {
        return null;
    }
    return $url;
}

// Proxy guard: returns the live upstream URL, or answers 503 + Retry-After and exits.
function simpleti_require_live_upstream(): string
{
    $file = simpleti_state_file();
    $url = $file === null ? null : simpleti_live_upstream(simpleti_read_state($file), time());
    if ($url === null) {
        http_response_code(503);
        header('Content-Type: application/json; charset=utf-8');
        header('Retry-After: ' . SIMPLETI_RETRY_AFTER_S);
        echo json_encode(['error' => ['type' => 'upstream_unavailable', 'message' => 'Simplicio 27B upstream is offline. Retry later.']]);
        exit;
    }
    return $url;
}
