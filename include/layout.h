
#ifndef __LAYOUT_H_INCLUDED__
#define __LAYOUT_H_INCLUDED__

#include <vector>
#include <map>
#include <utility>

using namespace std;

struct Terminal {
    Terminal(char t, int n) {type = t; num = n;}
    char type;
    int num;
};

struct Node {
    void add_terminal(Terminal *t) {terms.push_back(t);}
    ~Node();
    
    vector<Terminal*> terms;
};

class Transistor {
public:
    Transistor(int n);
    Terminal* D() {return terms[0];}
    Terminal* G() {return terms[1];}
    Terminal* S() {return terms[2];}
    int get_num() {return num;}
    
private:
    int num;
    vector<Terminal*> terms;
};

class Layout {
public:
    ~Layout();
    void add_node(Node *n);
    void add_transistor(int n, Transistor *t);
    void add_in(int n, Terminal *t);
    void add_out(int n, Terminal *t);

    Node* get_node(int n) {return nodes[n];}
    Transistor* get_transistor(int n) {return trans[n];}
    Terminal* get_in(int n) {return in[n];}
    Terminal* get_out(int n) {return out[n];}

    map<int, Transistor*>::iterator find_transistor(int n);
    map<int, Terminal*>::iterator find_in(int n);
    map<int, Terminal*>::iterator find_out(int n);
    
    bool is_transistor(int n);
    bool is_in(int n);
    bool is_out(int n);

    int nodes_size() {return nodes.size();}
    int trans_size() {return trans.size();}
    int in_size() {return in.size();}
    int out_size() {return out.size();}
    
private:
  vector<Node*> nodes;
  map<int, Transistor*> trans;
  map<int, Terminal*> in;
  map<int, Terminal*> out;
};
#endif 
