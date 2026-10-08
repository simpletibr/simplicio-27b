<?php
// GET /v1/status: estado público, sem segredos e sem a URL do túnel.
// deploy_sha vem de DEPLOY_SHA ao lado deste arquivo (git rev-parse HEAD > gateway/DEPLOY_SHA no deploy; não versionado).
header('Content-Type: application/json; charset=utf-8');
header('Cache-Control: no-store');
header('Access-Control-Allow-Origin: *');
if (($_SERVER['REQUEST_METHOD'] ?? 'GET') !== 'GET') {
    http_response_code(405);
    header('Allow: GET');
    echo json_encode(['error' => 'method not allowed']);
    exit;
}
$file = getenv('SIMPLETI_UPSTREAM_FILE') ?: '';
$state = is_file($file) ? json_decode((string) file_get_contents($file), true) : null;
$registered = is_array($state) && is_string($state['upstream_url'] ?? null) && $state['upstream_url'] !== '';
$updated = $registered && is_int($state['updated'] ?? null) ? $state['updated'] : null;
$shaFile = __DIR__ . '/DEPLOY_SHA';
$sha = is_file($shaFile) ? trim((string) file_get_contents($shaFile)) : '';
echo json_encode([
    'upstream_registered' => $registered,
    'updated_at' => $updated === null ? null : gmdate('Y-m-d\TH:i:s\Z', $updated),
    'deploy_sha' => preg_match('/^[0-9a-f]{40}$/', $sha) === 1 ? $sha : null,
]);
