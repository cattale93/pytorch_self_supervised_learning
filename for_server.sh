# Quick commit command to commit last changes to be run on GPU server
commit_msg=${1:-"for server"}
cd /home/ale/Documents/Python/13_Tesi_2/ || exit
git add --all
git commit -m "$commit_msg"
git push -u origin master
git status