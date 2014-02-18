PYTHON_INCLUDE = /Library/Frameworks/Python.framework/Versions/2.7/include/python2.7
BOOST_INCLUDE = /usr/local/Cellar/boost/1.55.0/include

BASE_LIB_DIR = /usr/local/lib
PYTHON_LIB_DIR = /Library/Frameworks/Python.framework/Versions/2.7/lib/

PYTHON_LIB = python2.7
BOOST_LIB = boost_python

TARGET = hello_world

ALL_LIB = -L$(BASE_LIB_DIR) -L$(PYTHON_LIB_DIR) -l$(BOOST_LIB) -l$(PYTHON_LIB)
ALL_INC = -I$(PYTHON_INCLUDE) -I$(BOOST_INCLUDE)

# .so file is the file that python will import as a module
$(TARGET).so: $(TARGET).o
	g++ -shared -Wl, $(TARGET).o $(ALL_LIB) -o $(TARGET).so

$(TARGET).o: $(TARGET).cpp
	g++ $(ALL_INC) -fPIC -c $(TARGET).cpp



