# Makefile for Latent Structural SVM

CC=g++ -Wall 
#CFLAGS= -g 
#CFLAGS= -O3 -fomit-frame-pointer -ffast-math
CFLAGS = -O0 -g
LD=g++ -Wall
LDFLAGS= $(CFLAGS)
ESSFLAGS = $(CFLAGS) -c
#LDFLAGS= -O3
#LDFLAGS = -O3 -pg
LIBS= -lm
MOSEK_ROOT = /afs/cs.stanford.edu/u/pawan/Project/mosek/6/tools/platform/linux64x86
MOSEK_H = $(MOSEK_ROOT)/h
MSKLIBPATH = $(MOSEK_ROOT)/bin
#MOSEK_H= /Users/rafiwitten/libs/mosek/6/tools/platform/osx64x86/h
#MSKLINKFLAGS= -lirc -lguide -limf -lsvml -lunwind -lmosek64 -lpthread -lc -ldl -lm
MSKLINKFLAGS= -lmosek64 -lpthread -lm
#MSKLIBPATH= /Users/rafiwitten/libs/mosek/6/tools/platform/osx64x86/bin
SFMTPATH= ./SFMT-src-1.3.3

all: ess svm_bbox_learn svm_bbox_classify

clean: 
	rm -f *.o
	rm -f svm_bbox_learn svm_bbox_classify ess

ess:    ess.cc quality_pyramid.cc quality_box.cc
	#g++ -O3 -o ess ess.cc quality_pyramid.cc quality_box.cc 
	#g++ -O3 -c -D__MAIN__ -o ess ess.cc quality_pyramid.cc quality_box.cc 
	g++ $(ESSFLAGS) quality_pyramid.cc 
	g++ $(ESSFLAGS) quality_box.cc 
	g++ $(ESSFLAGS) ess.cc 
	g++ $(ESSFLAGS) kernel.hh

svm_bbox_learn: svm_struct_latent_spl.o svm_common.o mosek_qp_primal_optimize.o mosek_qp_optimize.o svm_struct_latent_api.o SFMT.o
	$(LD) $(LDFLAGS) quality_pyramid.o quality_box.o ess.o svm_struct_latent_spl.o svm_common.o mosek_qp_optimize.o mosek_qp_primal_optimize.o SFMT.o svm_struct_latent_api.o -o svm_bbox_learn $(LIBS) -L $(MSKLIBPATH) $(MSKLINKFLAGS)

svm_bbox_classify: svm_struct_latent_classify.o svm_common.o svm_struct_latent_api.o SFMT.o
	$(LD) $(LDFLAGS) quality_pyramid.o quality_box.o ess.o svm_struct_latent_classify.o svm_common.o SFMT.o svm_struct_latent_api.o -o svm_bbox_classify $(LIBS)

svm_struct_latent_spl.o: svm_struct_latent_spl.c
	$(CC) -c $(CFLAGS) svm_struct_latent_spl.c -o svm_struct_latent_spl.o

svm_common.o: ./svm_light/svm_common.c ./svm_light/svm_common.h ./svm_light/kernel.h
	$(CC) -c $(CFLAGS) ./svm_light/svm_common.c -o svm_common.o

mosek_qp_primal_optimize.o: mosek_qp_primal_optimize.c
	$(CC) -c $(CFLAGS) mosek_qp_primal_optimize.c -o mosek_qp_primal_optimize.o -I $(MOSEK_H)

mosek_qp_optimize.o: mosek_qp_optimize.c
	$(CC) -c $(CFLAGS) mosek_qp_optimize.c -o mosek_qp_optimize.o -I $(MOSEK_H)

svm_struct_latent_api.o: svm_struct_latent_api.c svm_struct_latent_api_types.h svm_struct_latent_api.h
	$(CC) -c $(CFLAGS) svm_struct_latent_api.c -o svm_struct_latent_api.o

svm_struct_latent_classify.o: svm_struct_latent_classify.c
	$(CC) -c $(CFLAGS) svm_struct_latent_classify.c -o svm_struct_latent_classify.o

SFMT.o: 
	$(CC) -c -DMEXP=607 $(SFMTPATH)/SFMT.c -o SFMT.o
