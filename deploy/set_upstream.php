<?php
/**
 * Gateway helper for Simplicio 27B (issue #5).
 * Drop in as simpleti.com.br/api/set_upstream.php
 *
 * POST {"upstream_url":"https://….trycloudflare.com"} → store
 * POST {"clear": true} or {"upstream_url": null} or {"upstream_url": ""} → delete
 */
header('Content-Type: application/json; charset=utf-8');
$key = $_GET['key'] ?? '';
$expected = getenv('SIMPLETI_ADMIN_KEY') ?: '';
if ($expected === '' || !hash_equals($expected, $key)) {
    http_response_code(403);
    echo json_encode(['error' => 'Unauthorized']);
    exit;
}
$raw = file_get_contents('php://input') ?: '';
$body = json_decode($raw, true);
if (!is_array($body)) {
    $body = [];
}
$file = getenv('SIMPLETI_UPSTREAM_FILE') ?: (sys_get_temp_dir() . '/simpleti-upstream.json');
$clear = !empty($body['clear'])
    || (array_key_exists('upstream_url', $body)
        && ($body['upstream_url'] === null || $body['upstream_url'] === ''));
if ($clear) {
    if (is_file($file)) {
        unlink($file);
    }
    echo json_encode(['ok' => true, 'upstream' => null]);
    exit;
}
$url = $body['upstream_url'] ?? '';
if (!is_string($url) || !preg_match('#^https://[a-zA-Z0-9.-]+#', $url)) {
    http_response_code(400);
    echo json_encode(['error' => 'invalid upstream_url']);
    exit;
}
$dir = dirname($file);
if (!is_dir($dir)) {
    mkdir($dir, 0750, true);
}
file_put_contents($file, json_encode(['upstream_url' => rtrim($url, '/'), 'updated' => time()]));
echo json_encode(['ok' => true, 'upstream' => rtrim($url, '/')]);
