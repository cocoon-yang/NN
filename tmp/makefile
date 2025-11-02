############################## 
#  Neuron Network Test Case  
############################## 

#Target
PROGRAM = test.exe 


INC = ..\inc
LIBS = 

#Flags 
CPPOPT =  /GS /GL /W3 /Gy  /EHsc  /I $(INC)  /DEBUG /std:c++20 
LINKOPT =   


# Compiler 
cc = cl.exe

# Source files 
CPPSOURCES = main.cpp  neuron.cpp  connection.cpp  layer.cpp  neuronnet.cpp

CPPOBJECTS = $(CPPSOURCES:.cpp=.obj) 

all: $(PROGRAM)

$(PROGRAM): $(CPPOBJECTS)
	link.exe /OUT:$(PROGRAM)  $(LINKOPT)   $(CPPOBJECTS)  

main.obj:
	$(cc)   $(CPPOPT)  /c  ../src/main.cpp  

neuron.obj:
	$(cc)   $(CPPOPT)  /c  ../src/neuron.cpp  

connection.obj:
	$(cc)   $(CPPOPT)  /c  ../src/connection.cpp  

layer.obj:
	$(cc)   $(CPPOPT)  /c  ../src/layer.cpp  

neuronnet.obj:
	$(cc)   $(CPPOPT)  /c  ../src/neuronnet.cpp  


clean: 
	del   $(CPPOBJECTS)  $(PROGRAM)