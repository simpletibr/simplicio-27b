<?php
// Per-run gate token saved by set_upstream.php. The proxy sends it to the Colab gate.
function simpleti_upstream_token(string $file): ?string
{
    $state = is_file($file) ? json_decode((string) file_get_contents($file), true) : null;
    $token = is_array($state) ? ($state['upstream_token'] ?? null) : null;
    return is_string($token) && preg_match('/^[A-Za-z0-9_-]{32,128}$/D', $token) === 1 ? $token : null;
}
