#include <stdio.h>
#include <boost/python.hpp>

class HelloWorldClass{
    private:
        int private_data_member;
    public:
        HelloWorldClass();
        ~HelloWorldClass();
        void say_hello();
        int get_integer();
};

HelloWorldClass::HelloWorldClass(){
    std::cout << "Constructor called\n";
}

HelloWorldClass::~HelloWorldClass(){
    std::cout << "Destructor called\n";
}

void HelloWorldClass::say_hello(){
    std::cout<<"Hello World\n";
}

int HelloWorldClass::get_integer(){
    return 42;
}

using namespace boost::python;
BOOST_PYTHON_MODULE(hello_world) // this parameter needs to match filename
{
    class_<HelloWorldClass>("HelloWorldClass")
        .def("say_hello", &HelloWorldClass::say_hello)
        .def("get_integer", &HelloWorldClass::get_integer);
}
