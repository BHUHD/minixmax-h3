#!/usr/bin/env bash
# Source CANN toolkit env (default 9.1.0). Safe to source from other scripts.
# Usage: source "$(dirname "$0")/cann_env.sh"
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
fi

H3_CANN_VER="${H3_CANN_VER:-9.1.0}"

_h3_resolve_cann_home() {
  local ver="$1"
  local info home
  for info in /usr/local/Ascend/cann-*/aarch64-linux/ascend_toolkit_install.info; do
    [[ -f "${info}" ]] || continue
    if grep -q "^version=${ver}$" "${info}"; then
      home="$(dirname "$(dirname "${info}")")"
      local canon="/usr/local/Ascend/cann-${ver}"
      if [[ "${home}" != "${canon}" && ! -e "${canon}" ]]; then
        ln -sfn "${home}" "${canon}" 2>/dev/null || true
      fi
      if [[ -e "${canon}" ]]; then
        echo "${canon}"
      else
        echo "${home}"
      fi
      return 0
    fi
  done
  local home="/usr/local/Ascend/cann-${ver}"
  if [[ -f "${home}/set_env.sh" ]]; then
    echo "${home}"
    return 0
  fi
  if [[ -e /usr/local/Ascend/cann ]]; then
    local linked
    linked="$(readlink -f /usr/local/Ascend/cann)"
    if [[ -f "${linked}/set_env.sh" ]]; then
      echo "${linked}"
      return 0
    fi
  fi
  if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    if [[ -e /usr/local/Ascend/ascend-toolkit/latest ]]; then
      readlink -f /usr/local/Ascend/ascend-toolkit/latest
    else
      echo "/usr/local/Ascend/ascend-toolkit"
    fi
    return 0
  fi
  return 1
}

if ! H3_CANN_HOME="$(_h3_resolve_cann_home "${H3_CANN_VER}")"; then
  echo "[cann_env] CANN ${H3_CANN_VER} not found under /usr/local/Ascend" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090,SC1091
source "${H3_CANN_HOME}/set_env.sh"
export H3_CANN_HOME ASCEND_HOME_PATH="${H3_CANN_HOME}"
