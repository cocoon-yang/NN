############################## 
#  Neuron Network Test Case  
############################## 

#Target
PROGRAM = test.exe 


INC = ..\inc
LIBS = 

#Flags 
CPPOPT =  /GS /GL /W3 /Gy  /EHsc  /I $(INC)  /DEBUG
LINKOPT =   


# Compiler 
cc = cl.exe

# Source files 
CPPSOURCES = main.cc  nn.cc

CPPOBJECTS = $(CPPSOURCES:.cc=.obj) 

all: $(PROGRAM)

$(PROGRAM): $(CPPOBJECTS)
	link.exe /OUT:$(PROGRAM)  $(LINKOPT)   $(CPPOBJECTS)  

main.obj:
	$(cc)   $(CPPOPT)  /c  ../src/main.cc  

nn.obj:
	$(cc)   $(CPPOPT)  /c  ../src/nn.cc  

clean: 
	del   $(CPPOBJECTS)  $(PROGRAM)