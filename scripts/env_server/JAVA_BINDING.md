Java 绑定运行方法（Apptainer）

目的
- SciWorld 需要 Java。Apptainer 镜像内未必自带 JRE/JDK，需要把计算节点上的 Java 绑定进容器。
- 本目录脚本已统一使用 ~/uv_envs 下的环境。

前提
- 计算节点上可以找到 java（`command -v java` 有输出）。
- Apptainer 镜像：`~/containers/cuda_12.4.1-devel-ubuntu22.04.sif`

如果集群没有 Java 11（推荐）
1) 下载并安装 JDK 11 到 `$HOME/jdk-11`（一次即可）
```
JDK_HOME="$HOME/jdk-11"
curl -L --fail -o /tmp/temurin11.tar.gz \
  "https://api.adoptium.net/v3/binary/latest/11/ga/linux/x64/jdk/hotspot/normal/eclipse"
rm -rf "$JDK_HOME" && mkdir -p "$JDK_HOME"
tar -xzf /tmp/temurin11.tar.gz -C "$JDK_HOME" --strip-components=1
rm -f /tmp/temurin11.tar.gz
"$JDK_HOME/bin/java" -version
"$JDK_HOME/bin/javac" -version
```

2) 绑定 JDK 11 到容器，并强制 pyjnius 使用该 JVM
```
JAVA_HOME="$HOME/jdk-11"
APPTAINER_ARGS=(
  --nv
  --bind "$JAVA_HOME:$JAVA_HOME"
  --bind /usr/lib/jvm:/usr/lib/jvm
  --env JAVA_HOME="$JAVA_HOME"
  --env JRE_HOME="$JAVA_HOME"
  --env PATH="$JAVA_HOME/bin:$PATH"
  --env JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
)
```

步骤 1：获取 JAVA_HOME（在计算节点上执行）
```
JAVA_BIN=$(command -v java)
JAVA_REAL=$(readlink -f "$JAVA_BIN")
JAVA_HOME=$(dirname "$(dirname "$JAVA_REAL")")
```

步骤 2：用 Apptainer 运行（示例：SciWorld）
```
IMG="$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif"
APPTAINER_ARGS=(
  --nv
  --bind /usr/lib/jvm:/usr/lib/jvm
  --env JAVA_HOME="$JAVA_HOME"
  --env PATH="$JAVA_HOME/bin:$PATH"
)

apptainer exec "${APPTAINER_ARGS[@]}" "$IMG" \
  bash -lc "bash $HOME/projects/Word2World/scripts/env_server/start_sciworld.sh"
```

其他环境同理（alfworld/textworld/webshop）：
```
bash $HOME/projects/Word2World/scripts/env_server/start_alfworld.sh
bash $HOME/projects/Word2World/scripts/env_server/start_textworld.sh
bash $HOME/projects/Word2World/scripts/env_server/start_webshop.sh
```

注意事项
- 若你使用了 `--contain` 或 `--no-home`，导致容器内 HOME 不是宿主 HOME，
  需加 `--home "$HOME"` 或 `--bind "$HOME:$HOME"`，否则脚本找不到 `~/uv_envs`。
