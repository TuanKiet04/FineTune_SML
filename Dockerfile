FROM n8nio/n8n:latest

USER root

# Reinstall apk-tools (n8n đã xóa nó)
RUN ARCH=$(uname -m) && \
    wget -qO- "http://dl-cdn.alpinelinux.org/alpine/latest-stable/main/${ARCH}/" | \
    grep -o 'href="apk-tools-static-[^"]*\.apk"' | head -1 | cut -d'"' -f2 | \
    xargs -I {} wget -q "http://dl-cdn.alpinelinux.org/alpine/latest-stable/main/${ARCH}/{}" && \
    tar -xzf apk-tools-static-*.apk && \
    ./sbin/apk.static -X http://dl-cdn.alpinelinux.org/alpine/latest-stable/main \
        -U --allow-untrusted add apk-tools && \
    rm -rf sbin apk-tools-static-*.apk

# Cài FFmpeg và ImageMagick
RUN apk update && \
    apk add --no-cache \
        ffmpeg \
        imagemagick \
        font-noto \
        ttf-dejavu

# Cài npm packages (vẫn là root)
RUN npm install -g @qdrant/js-client-rest fluent-ffmpeg assemblyai @langchain/community typescript

# Chuyển về user node ở cuối
USER node