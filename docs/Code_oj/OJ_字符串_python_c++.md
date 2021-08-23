# 字符串输入输出

练习链接：[(2条未读私信) 牛客竞赛_ACM/NOI/CSP/CCPC/ICPC算法编程高难度练习赛_牛客竞赛OJ (nowcoder.com)](https://ac.nowcoder.com/acm/contest/5657)

## 1. 字符串 1

<img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210327110746351.png" alt="image-20210327110746351" style="zoom:67%;" />

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    int n;
    cin >> n;
    vector<string> vec(n);
    for (int i=0; i<n; i++) {
        cin >> vec[i];
    }
    sort(vec.begin(), vec.end());
    for (int i=0; i<n; i++) {
        cout << vec[i];
        if (i != n-1) cout << ' ';
    }
}

// 解2
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

int main(){
    int n;
    string str;
    cin >> n;
    vector<string> v;
    while(n){
        cin >> str;
        v.push_back(str);
        --n;
    }
    sort(v.begin(), v.end(), less<string>());
    
    for(int i = 0; i < v.size(); i++){
        cout << v[i] << " ";
    }
    
// less(小于)
// greater(大于)
// equal_to(等于)
// not_equal_to(不相等)
// less_equal(小于等于)
// greater_equal(大于等于)
    
    return 0;
}
```

```cpp
// 自写 2ms
#include <iostream> 
#include <string>
#include <vector>
using namespace std;

int main() {
    int n;
    string s;
    vector<string> vec;
    cin >> n;
    while (n--) {
        cin >> s;
        if (cin.get() == ' ' || cin.get() != '\n') 
            vec.push_back(s);
    }

    string tmp;
    for (int i=0; i<vec.size(); i++) {
        for (int j=i; j<vec.size(); j++) {
            if (vec[j] < vec[i]) {
                tmp = vec[j];
                vec[j] = vec[i];
                vec[i] = tmp;
            }
        }
    }
    
    for (int i=0; i<vec.size(); i++){
        if (i == vec.size()-1) {
            cout << vec[i] << endl;
        }
        else cout << vec[i] <<' ';
    }
}
```

```python
import sys
sz = int(input().strip())
print(' '.join(sorted(input().strip().split())))


% 解2
import sys
n1 = int(sys.stdin.readline())
n2 = list(map(str, sys.stdin.readline().split()))
n2.sort()
print(" ".join((n2)))

% 解3
input()
res=input().split()
res.sort()
res=' '.join(res)
print(res)

```

## 2. 字符串 2

![image-20210327114905301](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210327114905301.png)

```cpp
#include<bits/stdc++.h>
using namespace std;

int main() {
    vector<string> str;
    string s;
    while (cin >> s) {
        str.push_back(s);
        if (cin.get() == '\n'){
            sort(str.begin(), str.end());
            for (int i=0; i<str.size(); i++) {
                cout << str[i] << ' ';
            }
            cout << endl;
            str.clear();
        }
    }
}
```

```python
import sys

arr = sys.stdin.readlines()

for line in arr:
    vec = [x for x in line.strip().split()]
    vec.sort()
    print(" ".join(vec))
```

## 3. 字符串 3

![image-20210327115528056](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210327115528056.png)

```cpp
#include<bits/stdc++.h>
#include <sstream>
using namespace std;

int main(){
    vector<string> str;
    string s;
    string t;
    // 将 输入的字符串放入s
    while (getline(cin, s)) {
        stringstream ss(s); // 将s 构建成字符串流以供下面再次模拟输入
        while (getline(ss, t, ',')) {
            str.push_back(t);
        }
        
        sort(str.begin(), str.end());
        for (int i=0; i<str.size(); i++) {
            cout << str[i];
            if (i != str.size()-1) cout << ',';
        }
        cout << endl;
        str.clear();
        
    }
}
```

```python
while 1:
    try:
        s = input().split(",")
        temp = sorted(s)
        print(",".join(temp))
    except:
        break
        
% 解2
while True:
    try:
        lis = input().split(',')
        print(','.join(sorted(lis)))
    except:
        break
        
% 解3
import sys
line = sys.stdin.readlines()
for l in line:
    vec = [x for x in l.strip().split(",")]
    vec.sort()
    print(",".join(vec))
```

