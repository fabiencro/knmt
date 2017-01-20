import subprocess
import os

def get_installed_path():
    return os.path.dirname(os.path.realpath(__file__))

def get_current_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd = get_installed_path()).strip()
    except:
        return "**Could not retrieve git-hash"

def get_current_git_diff():
    try:
        return subprocess.check_output(['git', 'diff'], cwd = get_installed_path())
    except:
        return "**Could not retrieve git-diff"

def get_package_git_hash():
    try:
        import _build
        return _build.__build__
    except:
        return "**Could not get package git-hash**"

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
    print "package build diff (git diff):", get_package_git_diff()
    
    print "\n\n********** current version info ***********"
    print "git hash:", get_current_git_hash()
    print "git diff:"
    print get_current_git_diff()

if __name__ == "__main__":
    main()