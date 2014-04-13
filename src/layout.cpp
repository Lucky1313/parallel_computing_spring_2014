#include "layout.h"

using namespace std;

Node::~Node() {
    for (unsigned int i=0; i<terms.size(); ++i) {
        delete terms[i];
    }
}

//Positions of terminals are always D = (x, y-1), G = (x-1, y), S = (x, y+1)
Transistor::Transistor(int n) {
    num = n;
    Terminal* d = new Terminal('D', num);
    terms.push_back(d);
    Terminal* g = new Terminal('G', num);
    terms.push_back(g);
    Terminal* s = new Terminal('S', num);
    terms.push_back(s);
  }

//(0,0) is top left corner
Layout::~Layout() {
    for (unsigned int i=0; i<nodes.size(); ++i) {
	delete nodes[i];
    }
    for (unsigned int i=0; i<trans.size(); ++i) {
        delete trans[i];
    }
}

void Layout::add_node(Node *n) {
    nodes.push_back(n);
}

void Layout::add_transistor(int n, Transistor *t) {
    trans.insert(pair<int, Transistor*>(n, t));
}

void Layout::add_in(int n, Terminal *t) {
    in.insert(pair<int, Terminal*>(n, t));
}

void Layout::add_out(int n, Terminal *t) {
    out.insert(pair<int, Terminal*>(n, t));
}

map<int, Transistor*>::iterator Layout::find_transistor(int n) {
    return trans.find(n);
}

map<int, Terminal*>::iterator Layout::find_in(int n) {
    return in.find(n);
}

map<int, Terminal*>::iterator Layout::find_out(int n) {
    return out.find(n);
}

bool Layout::is_transistor(int n) {
    return (trans.find(n) != trans.end());
}

bool Layout::is_in(int n) {
    return (in.find(n) != in.end());
}

bool Layout::is_out(int n) {
    return (out.find(n) != out.end());
}

