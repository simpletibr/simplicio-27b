<?php
// Gateway helper for Simplicio 27B: simpleti.com.br/api/set_upstream.php
// POST only, header "Authorization: Bearer <SIMPLETI_ADMIN_KEY>".
// {"upstream_url":"https://<name>.trycloudflare.com"} stores; {"clear": true}, null or "" deletes.
header('Content-Type: application/json; charset=utf-8');

function respond(int $status, array $body): void
{
    http_response_code($status);
    echo json_encode($body);
    exit;
}

if (($_SERVER['REQUEST_METHOD'] ?? '') !== 'POST') {
    header('Allow: POST');
    respond(405, ['error' => 'Method Not Allowed']);
}
$expected = getenv('SIMPLETI_ADMIN_KEY') ?: '';
if (strlen($expected) < 32) {
    respond(500, ['error' => 'admin key not configured']);
}
$auth = $_SERVER['HTTP_AUTHORIZATION'] ?? $_SERVER['REDIRECT_HTTP_AUTHORIZATION'] ?? '';
$key = (is_string($auth) && strncmp($auth, 'Bearer ', 7) === 0) ? substr($auth, 7) : '';
if (!hash_equals($expected, $key)) {
    header('WWW-Authenticate: Bearer');
    respond(401, ['error' => 'Unauthorized']);
}
$body = json_decode(file_get_contents('php://input') ?: '', true);
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
    respond(200, ['ok' => true, 'upstream' => null]);
}
$url = $body['upstream_url'] ?? '';
if (!is_string($url) || !preg_match('#^https://[a-z0-9-]+\.trycloudflare\.com$#D', $url)) {
    respond(400, ['error' => 'invalid upstream_url']);
}
$dir = dirname($file);
if (!is_dir($dir)) {
    mkdir($dir, 0750, true);
}
$tmp = tempnam($dir, '.upstream-');
$json = json_encode(['upstream_url' => $url, 'updated' => time()]);
if ($tmp === false
    || file_put_contents($tmp, $json, LOCK_EX) === false
    || !chmod($tmp, 0644)
    || !rename($tmp, $file)) {
    if ($tmp !== false && is_file($tmp)) {
        unlink($tmp);
    }
    respond(500, ['error' => 'cannot write upstream file']);
}
respond(200, ['ok' => true, 'upstream' => $url]);
