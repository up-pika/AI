# 判题系统状态   

等待评测: 评测系统还没有评测到这个提交，请稍候  

正在评测: 评测系统正在评测，稍候会有结果  

编译错误:您提交的代码无法完成编译，点击“编译错误”可以看到编译器输出的错误信息  

答案正确: 恭喜！您通过了这道题   

运行错误: 您提交的程序在运行时发生错误,可能是空指针  

部分正确: 您的代码只通过了部分测试点，继续努力！  

格式错误: 您的程序输出的格式不符合要求（比如空格和换行与要求不一致）  

答案错误: 您的程序未能对评测系统的数据返回正确的结果   

运行超时: 您的程序未能在规定时间内运行结束  

内存超限: 您的程序使用了超过限制的内存  

异常退出: 您的程序运行时发生了错误  

返回非零: 您的程序结束时返回值非 0，如果使用 C 或 C++ 语言要保证 int main 函数最终 return 0  

浮点错误: 您的程序运行时发生浮点错误，比如遇到了除以 0 的情况  

 段错误 : 您的程序发生段错误，可能是数组越界，堆栈溢出（比如，递归调用层数太多）等情况引起  

多种错误: 您的程序对不同的测试点出现不同的错误   

内部错误:   请仔细检查你的代码是否有未考虑到的异常情况，例如非法调用、代码不符合规范等。

牛客oj：[(2条未读私信) 牛客竞赛_ACM/NOI/CSP/CCPC/ICPC算法编程高难度练习赛_牛客竞赛OJ (nowcoder.com)](https://ac.nowcoder.com/acm/contest/5657)



# 数组的输入输出

## 1. A+B 1

![image-20210823134116395](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210823134116395.png)

```c++
# include<iostream>
using namespace std;

int main(){
    int a, b;
    while (cin >> a >> b){
        cout << a + b << endl;
    }
    return 0;
}

// scanf & printf
#include<bits/stdc++.h>

int main(){
    int a, b;
    while(scanf("%d %d", &a, &b) != EOF){
        printf("%d\n",a+b);
    }
    return 0;
}
```

```python
import sys 
for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))
```

## 2. A+B 2

![image-20210327111601395](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210327111601395.png)

```cpp
# include<iostream>
using namespace std;

int main(){
    int t, a, b;
    cin >> t;
    while (t--){
        cin >> a >> b;
        cout << a+b << endl;
    }
}

// 解法2
# include<iostream>
# include<vector>

using namespace std;
int t, tmp;
int main(){
    cin >> t;
    vector<vector<int> > vec(t, vector<int>(2));
    for(int i=0; i<t; i++){
        for(int j=0; j<2; j++){
            cin >> vec[i][j];
            tmp += vec[i][j];
        }
        cout << tmp << endl;
        tmp = 0;
    }
    return 0;
}
```

```python
Q = []
X = input()
Q.append(X)
K = int(Q[0])
for v in range(0, K, 1):
    Y = input()
    Q.append(Y)
for i in range(1, K + 1, 1):
    D = Q[i].split(' ')
    E = [int(j) for j in D]
    s = sum(E)
    print(s)
```

## 3. A+B 3

![image-20210823134010703](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210823134010703.png)

```c++
#include<iostream>
using namespace std;
int main(){
    int a,b;
    while(cin >> a >> b && a != 0 && b != 0){
        cout << a + b << endl;
    }
}

// 解法1-1
 # include<iostream>
using namespace std;

int main() {
    int a, b;
    while (cin >> a >> b) {
        if (a == 0 && b == 0) break;
        cout<< a+b << endl;
    }
}

// 解法2
#include<iostream>
using namespace std;

int main(){
    int a, b;
    while(cin >> a >> b){
        if(a <= 0 || b <= 0 || b > 1e9 || a > 1e9) break;
        cout<< a + b << endl;
    }
    return 0;
}


// 解法3
#include<bits/stdc++.h>
#include<sstream>
using namespace std;
 
int main(){
    string s;
    while(getline(cin, s)){
        stringstream ss;
        ss << s;
        int num1 = 0, num2 = 0;
        ss >> num1;
        ss >> num2;
        if(num1 == 0 && num2 == 0)
            break;
        cout << num1 + num2 << endl;
    }
    return 0;
}

// scanf,printf 解法 与cin和cout运行速度持平，内存少点点
#include<iostream>
using namespace std;
int main(){
    int a,b;
    while(true){
        scanf("%d%d",&a,&b);
        if(a == 0 && b == 0)
            break;
        else
            printf("%d\n",a+b);
    }
    return 0;
}

#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;

int main() {
	int n;
	scanf("%d", &n);
	vector<int> a;
	vector<int> b;
	for(int i=0; i<n; i++) {
		int temp1, temp2;
		scanf("%d %d", &temp1, &temp2);
		a.emplace_back(temp1);
		b.emplace_back(temp2);
	}

	for (int i = 0; i < n; i++) {
		printf("%d\n", a[i] + b[i]);
		printf("hello");
	}
	return 0;
}
```

## 4. A+B 4

![image-20210823134717527](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210823134717527.png)

```c++
# include<iostream>
using namespace std;

int main() {
    int num, a;
    while (cin >> num) {
        if (num == 0) break;
        int sum = 0;
        while (num--){
        cin >> a;
        sum += a;
        }
        cout << sum << endl;
    }
}

// scanf & printf
# include<stdio.h>
using namespace std;

int main(){
    int n;
    while(scanf("%d", &n) != EOF){
        if(n == 0) break;
        int num, sum = 0; // sum=0一定要赋值，否则不能通过
        for(int i=0; i<n; i++){
            scanf("%d", &num);
            sum += num;
        }
        printf("%d\n", sum);
    }
    return 0;
}

// 解法2
#include<iostream>
using namespace std;
int main(){
    int n, a;
    while(true){
        scanf("%d",&n);
        if(n==0) break;
        int sum=0;
        for(int i=0;i<n;i++){
            scanf("%d",&a);
            sum+=a;
        }
        printf("%d\n",sum);
    }
    return 0;
}

// 解法3
#include<iostream>
using namespace std;

int main(){
    int n;
    scanf("%d", &n);
    while(n != 0){
        int sum = 0;
        while(n--){
            int k;
            scanf("%d", &k);
            sum += k;
        }
        printf("%d\n", sum);
        scanf("%d", &n);
    }
    return 0;
}
```

```python
import sys

for line in sys.stdin:
    alist = [int(i) for i in line.split(" ")]
    n = alist[0]
    if n == 0:
        break
    print(sum(alist[1:]))
    
% 解1
import sys
for line in sys.stdin:
    a = list(map(int, line.split(' ')))
    if a[0] == 0:
        break
    print(sum(a[1:]))
```

## 5. A+B 5 

![image-20210327111731464](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210327111731464.png)

```C++
# include<iostream>
using namespace std;

int main(){
    int t, a;
    cin >> t;
    while(t--){
        int num, sum = 0;
        cin >> num;
        for(int i=0; i<num; i++){
            cin >> a;
            sum += a;
        }
        cout << sum << endl;
    }
    return 0;
}

// 解2
#include<iostream>

using namespace std;

int main(){
    int t;
    scanf("%d",&t);
    while(t--){
        int n;
        scanf("%d",&n);
        int sum = 0;
        while(n--){
            int k;
            scanf("%d",&k);
            sum += k;
        }
        printf("%d\n",sum);
    }
    return 0;
}

// 解3
#include<stdio.h>
#include<string.h>
#include<ctype.h>
int main()
{
    int n;
    while(scanf("%d",&n)!=EOF)
    {
        for(int j=0;j<n;j++)
        {
            int sum=0;
            int temp;
            int num;
            scanf("%d", &num);
            for(int i=0; i<num; i++)
            {
                scanf("%d",&temp);
                sum+=temp;
            }
            printf("%d\n",sum);

        }

    }
}

```

```python
n = int(input())
for i in range(n):
    lis = list(map(int, input().split()))
    print(sum(lis[1:]))
```

## 6. A+B 6

![image-20210823145130800](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210823145130800.png)

```CPP
#include <iostream>
using namespace std;

int main() {
    int n, a;
    while (cin >> n){
        int sum = 0;
        while (n--) {
            cin >> a;
            sum += a;
            }
        cout << sum << endl;
        }
    }

// 解2
#include <iostream>
using namespace std;

int main() {
    int N, sum, tmp;
    while(scanf("%d", &N) != EOF) {
        sum = 0;
        for(int n = 0; n < N; ++ n) {
            scanf("%d", &tmp);
            sum += tmp;
        }
        printf("%d\n", sum);
    }
    
    return 0;
}
```

```python
class Solution:
    def summ(self, a):
        return sum(a[1:])
    
if __name__ == "__main__":
    while True:
        try:
            a=list(map(int, input().split(' ')))
            print(Solution().summ(a))
        except:
            break
            
% 解2
import sys 

while True:
    line=sys.stdin.readline()
    if not line: break
    line = line.split()
    n = int(line[0])
    s = sum(map(int, line[1:]))
    print(s) 
```

## 7. A+B 7

<img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210327103002566.png" alt="image-20210327103002566" style="zoom:67%;" />

```cpp
#include <iostream> 
using namespace std;

int main() {
    int n, sum = 0;
    while (cin >> n) {
        sum += n;
        if (cin.get() == '\n') {
            cout<< sum << endl;
            sum = 0;
        }
    }
}
```

```python
import sys

for line in sys.stdin:
    res=list(map(int, line.split( )))
    print(sum(res))
    
% 解2
while True:
    try:
        A = list(map(int, input().split(" ")))
        print(sum(A))
    except:break
        
% 解3
data = []
while True:
    try:
        temp = list(map(int, input().split(' ')))
        data.append(temp)
    except:
        break
for item in data:
    print(sum(item))
    
% 解3
import sys
for line in sys.stdin:
    L = line.split()
    res = 0
    for x in L:
        res += int(x)
    print(res)
```



