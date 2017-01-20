import subprocess
import os

def get_installed_path():
    return os.path.dirname(os.path.realpath(__file__))

def get_current_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                       cwd = get_installed_path(),
                                       stderr = subprocess.STDOUT).strip()
    except:
        return "**Could not retrieve git-hash**"

def get_current_git_diff():
    try:
        return subprocess.check_output(['git', 'diff'], 
                                       cwd = get_installed_path(),
                                       stderr = subprocess.STDOUT)
    except:
        return "**Could not retrieve git-diff**"
    
    
def is_current_git_dirty():
    try:
        DEVNULL = open(os.devnull, 'wb')
        returncode =  subprocess.call(['git', 'diff-index', '--quiet', 'HEAD', '--'], 
                                      cwd = get_installed_path(), 
                                      stdout = DEVNULL,
                                      stderr = DEVNULL)
        if returncode == 0:
            return "clean"
        elif returncode == 1:
            return "dirty"
        else:
            return "unknown"
    except:
        return "unknown"


def get_package_git_hash():
    try:
        import _build
        return _build.__build__
    except:
        return "**Could not get package git-hash**"

def get_package_dirty_status():
    try:
        import _build
        return _build.__dirty_status__
    except:
        return "**Could not get package git dirty status**"

def get_package_git_diff():
    try:
        import _build
        import json
        return _build.__git_diff__
    except:
        return "**Could not get package git-diff**"

def main(options = None):
    import _version
    print "package version:", _version.__version__
    print "installed in:", get_installed_path()
    
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
