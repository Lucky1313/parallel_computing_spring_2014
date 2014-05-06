
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "layout.h"
#include "ga.h"

using namespace std;

//Read input file to extract node data
void read_input(Layout *layout, char ifile[]) {
  ifstream file;
  file.open(ifile);
  string line;
  int node_count = 0;

  while (getline(file, line)) {
    istringstream iss(line);
    string tmp;

    Node* node = new Node();
    while (iss >> tmp) {

      Terminal* t;
      if (tmp[0] == 'T') {
	int num = tmp[1] - '0';
	
	Transistor* trans;
	if (!layout->is_transistor(num)) {
	  trans = new Transistor(num);
	  layout->add_transistor(num, trans);
	}
	else {
	  trans = layout->get_transistor(num);
	}
	switch(tmp[2]) {
	case 'D':
	  t = trans->D();
	  break;
	case 'G':
	  t = trans->G();
	  break;
	case 'S':
	  t = trans->S();
	  break;
	default:
	  cout << "No terminal error" << endl;
	  t = NULL;
	}
      }
      else if (tmp[0] == 'I') {
	t = new Terminal('I', tmp[1] - '0');
	layout->add_in(tmp[1], t);
      }
      else if (tmp[0] == 'O') {
	t = new Terminal('O', tmp[1] - '0');
	layout->add_out(tmp[1], t);
      }
      else if (tmp == "PWR") {
	t = new Terminal('P', 0);
      }
      else if (tmp == "GND") {
	t = new Terminal('Z', 0);
      }
      else {
	cout << "Error" << endl;
	t = NULL;
      }
      node->add_terminal(t);
    }
    layout->add_node(node);
    ++node_count;
  }
  
  cout << "Read " << layout->trans_size() << " transistors" << endl;
  cout << "As " << layout->nodes_size() << " nodes" << endl;
}

int main(int argc, char *argv[]) {
  Layout* main_layout = new Layout();
  if (argc >= 2) {
    cout << "Inputting file" << endl;
  }
  else {
    cout << "Improper number of arguments" << endl;
    return 1;
  }
  
  read_input(main_layout, argv[1]);

  launch_ga(main_layout);
  return 0;
}
