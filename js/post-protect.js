(function() {
    'use strict';

    function b64ToBytes(base64) {
        var binary = window.atob(base64);
        var bytes = new Uint8Array(binary.length);

        for (var i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }

        return bytes;
    }

    function concatBytes(a, b) {
        var merged = new Uint8Array(a.length + b.length);
        merged.set(a, 0);
        merged.set(b, a.length);
        return merged;
    }

    async function decryptPayload(payload, password) {
        var encoder = new TextEncoder();
        var passwordKey = await window.crypto.subtle.importKey(
            'raw',
            encoder.encode(password),
            'PBKDF2',
            false,
            ['deriveKey']
        );

        var key = await window.crypto.subtle.deriveKey(
            {
                name: 'PBKDF2',
                salt: b64ToBytes(payload.salt),
                iterations: payload.iter,
                hash: 'SHA-256'
            },
            passwordKey,
            {
                name: 'AES-GCM',
                length: 256
            },
            false,
            ['decrypt']
        );

        var encrypted = concatBytes(b64ToBytes(payload.data), b64ToBytes(payload.tag));
        var decrypted = await window.crypto.subtle.decrypt(
            {
                name: 'AES-GCM',
                iv: b64ToBytes(payload.iv)
            },
            key,
            encrypted
        );

        return new TextDecoder().decode(decrypted);
    }

    function initProtectedPost(container) {
        var input = container.querySelector('.post-protect-input');
        var button = container.querySelector('.post-protect-button');
        var error = container.querySelector('.post-protect-error');
        var content = container.querySelector('.post-protect-content');
        var payloadText = container.getAttribute('data-protected-payload');
        var payload = null;

        if (!input || !button || !error || !content || !payloadText) {
            return;
        }

        try {
            payload = JSON.parse(payloadText);
        } catch (e) {
            error.textContent = '加密数据损坏，无法解锁。';
            return;
        }

        var unlock = async function() {
            var password = input.value || '';
            error.textContent = '';

            if (!password.trim()) {
                error.textContent = '请输入密码。';
                input.focus();
                return;
            }

            button.disabled = true;
            button.textContent = '解锁中...';

            try {
                var html = await decryptPayload(payload, password);
                content.innerHTML = html;
                content.hidden = false;
                container.classList.add('is-unlocked');

                if (window.MathJax && typeof MathJax.typesetPromise === 'function') {
                    MathJax.typesetPromise([content]).catch(function(err) {
                        console.error('MathJax typeset failed:', err);
                    });
                }
            } catch (e) {
                error.textContent = '密码错误，或文章数据无法解密。';
            } finally {
                button.disabled = false;
                button.textContent = '解锁';
            }
        };

        button.addEventListener('click', unlock);
        input.addEventListener('keydown', function(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                unlock();
            }
        });
    }

    document.addEventListener('DOMContentLoaded', function() {
        var containers = document.querySelectorAll('[data-protected-post]');

        for (var i = 0; i < containers.length; i++) {
            initProtectedPost(containers[i]);
        }
    });
})();
