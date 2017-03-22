import subprocess
import os
from collections import OrderedDict


def get_installed_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_current_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=get_installed_path(),
                                       stderr=subprocess.STDOUT).strip()
    except BaseException:
        return None


def get_current_git_diff():
    try:
        return subprocess.check_output(['git', 'diff'],
                                       cwd=get_installed_path(),
                                       stderr=subprocess.STDOUT)
    except BaseException:
        return None


def is_current_git_dirty():
    try:
        DEVNULL = open(os.devnull, 'wb')
        returncode = subprocess.call(['git', 'diff-index', '--quiet', 'HEAD', '--'],
                                     cwd=get_installed_path(),
                                     stdout=DEVNULL,
                                     stderr=DEVNULL)
        if returncode == 0:
            return "clean"
        elif returncode == 1:
            return "dirty"
        else:
            return "unknown"
    except BaseException:
        return "unknown"


def get_chainer_infos():
    try:
        import chainer
        result = OrderedDict([
            ("version", chainer.__version__),
            ("cuda", chainer.cuda.available),
            ("cudnn", chainer.cuda.cudnn_enabled),
        ])
        if chainer.cuda.available:
            try:
                import cupy
                cuda_version = cupy.cuda.runtime.driverGetVersion()
            except BaseException:
                cuda_version = "unavailable"
            result["cuda_version"] = cuda_version
        else:
            result["cuda_version"] = "unavailable"

        if chainer.cuda.cudnn_enabled:
            try:
                cudnn_version = chainer.cuda.cudnn.cudnn.getVersion()
            except BaseException:
                cudnn_version = "unavailable"
            result["cudnn_version"] = cudnn_version
        else:
            result["cudnn_version"] = "unavailable"

    except ImportError:
        result = OrderedDict([
            ("version", "unavailable"),
            ("cuda", "unavailable"),
            ("cudnn", "unavailable"),
            ("cuda_version", "unavailable"),
            ("cudnn_version", "unavailable")
        ])

    return result


def get_package_git_hash():
    try:
        import nmt_chainer._build
        return nmt_chainer._build.__build__
    except BaseException:
        return None


def get_package_dirty_status():
    try:
        import nmt_chainer._build
        return nmt_chainer._build.__dirty_status__
    except BaseException:
        return "unknown"


def get_package_git_diff():
    try:
        import nmt_chainer._build
#         import json
        return nmt_chainer._build.__git_diff__
    except BaseException:
        return None


def get_version_dict():
    import nmt_chainer._version
    result = OrderedDict({"package_version": nmt_chainer._version.__version__})
    current_git_hash = get_current_git_hash()
    if current_git_hash is not None:
        result["git"] = current_git_hash
        current_git_status = is_current_git_dirty()
        result["dirty_status"] = current_git_status
        if current_git_status == "dirty":
            result["diff"] = get_current_git_diff()
        result["version_from"] = "git call"
    else:
        package_git_hash = get_package_git_hash()
        if package_git_hash is not None:
            result["git"] = package_git_hash
            current_git_status = get_package_dirty_status()
            result["dirty_status"] = current_git_status
            if current_git_status == "dirty":
                result["diff"] = get_package_git_diff()
            result["version_from"] = "setup info"
        else:
            result["git"] = "unavailable"

    result["chainer"] = get_chainer_infos()
    return result


def main(options=None):
    import nmt_chainer._version
    print "package version:", nmt_chainer._version.__version__
    print "installed in:", get_installed_path()

    print "\n*********** chainer version ***********"
    chainer_infos = get_chainer_infos()
    for keyword in "version cuda cudnn cuda_version cudnn_version".split():
        print keyword, chainer_infos[keyword]

    print "\n\n********** package build info ***********"
    print "package build (git hash):", get_package_git_hash()
    package_dirty_status = get_package_dirty_status()
    if package_dirty_status == "clean":
        print "  - package git index is clean"
    elif package_dirty_status == "dirty":
        print "  - package git index is dirty"
        print "\npackage build diff (git diff):\n", get_package_git_diff()

    print "\n\n********** current version info ***********"
    print "git hash:", get_current_git_hash()

    current_dirty_status = is_current_git_dirty()

    if current_dirty_status == "clean":
        print "  - git index is clean"
    elif current_dirty_status == "dirty":
        print "  - git index is dirty"
        print "\ngit diff:\n"
        print get_current_git_diff()


if __name__ == "__main__":
    main()
