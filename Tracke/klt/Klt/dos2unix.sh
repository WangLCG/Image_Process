#!/bin/sh
#Need to install dos2unix and enconv software
#使用：放到源码根目录然后执行，该脚本会将该目录下（包括子目录）所有.cpp/.c/.h文件转化为UNIX格式，文件编码转化为UTF-8 无BOM格式，TAB键转化为4个空格
#最后会输出一个log文件，该文件记录的是文件编码被转化的文件列表
#


echo "\033[31m Begin dos2unix convert process....\033[0m"
 find . -regex '.*\.cpp\|.*\.h\|.*\.c\|' | xargs unix2dos -q    # dos to unix quiet mode

 FILE=$(find . -regex '.*\.cpp\|.*\.h\|.*\.c\|')
 
 for filename in $FILE
 do
    sed -i 's/\t/    /g' $filename   # tab to space
 done
echo "\033[31m End dos2unix convert process.\t You can see the result in ChangedCodeFileList.log \033[0m"

