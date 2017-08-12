/*************************************
 *
 *  Author: Tian-Li Yu
 *  Email: tianliyu@ntu.edu.tw
 *  Date: 2014 Dec
 *
 *
 *  Use array to simulate a double linked list
 *  Insert: O(1)
 *  Delete: O(1)
 *  Has: O(1)
 *  Space Complexity: O(ell)
**************************************/


#ifndef _DLLA_
#define _DLLA_

#include <cstdio>
#include <vector>
using namespace std;


class DLLA {

public:

    DLLA(int _length) : data(_length,false), pre(_length), next(_length) {
        tableSize = _length;
        elementSize = 0;
        head = -1;
    }

    void operator= (const DLLA& h) {
        head = h.head;
        elementSize = h.elementSize;
        data = h.data;
        pre = h.pre;
        next = h.next;
    }

    bool find(int key) const {
        return data[key];
    }

    void insert(int key) {

        ++elementSize;

        if (head == -1) {
            head = key;
            data[key] = true;
            pre[key] = key;
            next[key] = key;
        } else {
            data[key] = true;
            int temp = pre[head];
            pre[head] = key;
            pre[key] = temp;
            next[key] = head;
            next[pre[key]] = key;
        }
    }

    void erase(int key) {

        if (elementSize == 0) return;

        --elementSize;

        if (elementSize == 0) {
            data[key] = false;
            head = -1;
        } else {
            data[key] = false;
            next[pre[key]] = next[key];
            pre[next[key]] = pre[key];
            if (head == key)
                head = pre[key];
        }
    }

    void eraseAll() {

        head = -1;

        for (int i=0; i<tableSize; i++)
            data[i] = false;
    }

    int getElementSize() const {
        return elementSize;
    }

    bool isEmpty() const {
        return (elementSize==0);
    }

    // Iterator

    class iterator {
    protected:
        DLLA* pArray;
        int index;
        int real_end;
        bool flag;


    public:
        iterator(DLLA* p = NULL, int nIndex = -1) {
            pArray = p;
            index = nIndex;
            flag = false;
            if (pArray->head == -1)
                real_end = -1;
            else
                real_end = pArray->pre[pArray->head];
        }

        void operator++() {  // Prefix
            if (flag)
                index = -1;
            else {
                index = pArray->next[index];
                if (index == real_end)
                    flag = true;
                else if (index == pArray->head) index = -1;
            }
        }

        void operator++(int) {  // Postfix
            operator++();
        }

        bool operator==(const iterator& iter) const {
            return (index == iter.index);
        }

        bool operator!=(const iterator& iter) const {
            return !(operator==(iter));
        }

        int& operator*() {
            return index;
        }
    };

    iterator begin() {
        iterator iter(this, head);
        return iter;
    }

    iterator end() {
        iterator iter(this, -1);
        return iter;
    }

protected:
    int tableSize;
    int elementSize;
    vector<bool> data;
    vector<int> pre;
    vector<int> next;
    int head;

};


#endif
