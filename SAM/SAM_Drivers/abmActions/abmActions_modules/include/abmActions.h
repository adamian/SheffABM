// -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-

/*
* Copyright (C) 2015 WYSIWYD Consortium, European Commission FP7 Project ICT-612139
* Authors: Uriel Martinez
* email:   uriel.martinez@sheffield.ac.uk
* Permission is granted to copy, distribute, and/or modify this program
* under the terms of the GNU General Public License, version 2 or any
* later version published by the Free Software Foundation.
*
* A copy of the license can be found at
* $WYSIWYD_ROOT/license/gpl.txt
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
* Public License for more details
*/

#ifndef __ABMACTIONS_H__
#define __ABMACTIONS_H__

/*
* @ingroup icub_module
*
* \defgroup modules speechInteraction
*
* Receives recognised words and triggers the corresponding behaviour
*
* \author Uriel Martinez
*
* Copyright (C) 2015 WYSIWYD Consortium\n
* CopyPolicy: Released under the terms of the GNU GPL v2.0.\n
*
*/


#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <yarp/sig/all.h>
#include <yarp/os/all.h>
#include <yarp/dev/all.h>
#include <ctime>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include "wrdac/clients/icubClient.h"
#include "wrdac/subsystems/subSystem.h"
#include "wrdac/knowledge/object.h"
#include <db/PostgreSQL.h>
#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace yarp::dev;
using namespace std;
using namespace wysiwyd::wrdac;


class abmActions: public RFModule
{
    private:
        ICubClient *iCub;
        bool ABMconnected;
        SubSystem_ABM* SubABM;

        DataBase<PostgreSql>* ABMDataBase;
        Port speakOutputPort;
        
        // connection to database
        std::string server;
        std::string user;
        std::string password;
        std::string dataB;
        std::string savefile;

	    Port rpcPort;
        std::string m_masterName;

        Port rpc;
        string moduleName;

        string agentName;
        string resume;
        int rememberedInstance;
        int instance;

        Bottle personsNames;
        vector<Bottle> personsOPCID;
        Bottle defaultPersonsOPCID;
        Bottle currentPersonsOPCID;
        string personName;

        bool speak_actions;
        bool newActionProcess;
        int configNumberOfPersons;

    public:
        abmActions();
        ~abmActions();
        bool updateModule();
        bool configure(ResourceFinder &);
        bool interruptModule();
        double getPeriod();
        bool close();
        string currentDateTime();
        void prepareDataBase();
        void getPersonsIDs();
        void sendDataToABM(Bottle);
        bool getDataFromABM(Bottle);
        Bottle emptyTables();
};

#endif // __ABMACTIONS_H__

//----- end-of-file --- ( next line intentionally left blank ) ------------------

