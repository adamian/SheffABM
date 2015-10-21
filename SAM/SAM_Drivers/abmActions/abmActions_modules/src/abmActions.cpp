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


#include "abmActions.h"

abmActions::abmActions()
{
    m_masterName = "abmActions";
    instance = 0;
    newActionProcess = false;
}

abmActions::~abmActions()
{
}

bool abmActions::updateModule()
{
    Bottle cmd;
    Bottle response;
    
    rpcPort.read(cmd, true);
        
    if( boost::iequals(cmd.get(0).asString(),"start_action") )
    {
        // Prepares instance and opcid for the person (face) recognised to store faces and actions.
        cout << "STATE: start_action" << endl;
        response.add("ACK: STATE start_action");

        newActionProcess = true;
    }
    else if( boost::iequals(cmd.get(0).asString(),"store_action") )
    {
        // Stores actions done by the person.
        // The face recognised at the beginning of the process is keep until a new "start_action" command is send by the human.
        cout << "STATE: store_action" << endl;
        response.add("ACK: STATE store_action");
        sendDataToABM(cmd);        
    }
    else if( boost::iequals(cmd.get(0).asString(),"ask_action") )
    {
        // Retrieves actions done previously.
        // The face of the person sending the "ask_action" is first recognised in order to retrieved his/her actions.
        cout << "STATE: ask_action" << endl;
        response.add("ACK: STATE ask_action");
        
        getDataFromABM(cmd);        
    }
    else
    {
        // Commands not received from human.
        cout << "Unknown command" << endl;
        response.add("NACK: unknown command");
    }
 
    rpcPort.reply(response);
 
    Time::delay(2);

    return true;
}

bool abmActions::configure(ResourceFinder &rf)
{
    moduleName = rf.check("name", Value("abmActions"), "module name (string)").asString();
    
    setName(moduleName.c_str());

    Property config;
    config.fromConfigFile(rf.findFile("from").c_str());

    Bottle &nGeneral = config.findGroup("number_of_persons");
    configNumberOfPersons = nGeneral.find("n_persons").asInt();

    cout << "Number of persons: " << configNumberOfPersons << endl;

    Bottle &configPersonsNames = config.findGroup("names_list");
    string findPerson = "person_";
    ostringstream convert;
    
    cout << "Names list" << endl;
    for( int i = 0; i < configNumberOfPersons; i++ )
    {
        convert << (i+1);
        findPerson = findPerson + convert.str();
        personsNames.add(configPersonsNames.find(findPerson).asString().c_str());
        cout << findPerson << ": " << configPersonsNames.find(findPerson).asString().c_str() << endl;
        findPerson = "person_";
        convert.str("");
    }

    Bottle &configOPCIDList = config.findGroup("opcid_list");
    string findOPCID = "opcid_";
    convert.str("");
    
    cout << "OPCID list" << endl;
    for( int i = 0; i < configNumberOfPersons; i++ )
    {
        convert << (i+1);
        findOPCID = findOPCID + convert.str();
        defaultPersonsOPCID.add(configOPCIDList.find(findOPCID).asInt());
        cout << findOPCID << ": " << configOPCIDList.find(findOPCID).asInt() << endl;
        findOPCID = "opcid_";
        convert.str("");
    }
    
    SubABM = new SubSystem_ABM("from_ABM_INTERACTION");
    ABMconnected = (SubABM->Connect());
    std::cout << ((ABMconnected) ? "ABM_INTERACTION connected to ABM" : "ABM_INTERACTION didn't connect to ABM") << std::endl;


    bool isRFVerbose = false;
    iCub = new ICubClient(moduleName, "abmActions", "client.ini", isRFVerbose);
    iCub->opc->isVerbose &= false;
    if (!iCub->connect())
    {
        cout << "iCubClient : Some dpeendencies are not running..." << endl;
        Time::delay(1.0);
    }

    bool rpcPortOpen = rpcPort.open(("/" + moduleName + "/rpc").c_str());
    if( !rpcPortOpen )
	{
		cout << "Could not open ports. Exiting" << endl;
		return false;
	}


    if (!iCub->getABMClient())
    {
        cout << "WARNING ABM NOT CONNECTED" << endl;
    }

    rememberedInstance = rf.check("rememberedInstance", Value(1333)).asInt();
    agentName = rf.check("agentName", Value("Uriel")).asString().c_str();

    //conf group for database properties
    Bottle &bDBProperties = rf.findGroup("database_properties");
    server = bDBProperties.check("server", Value("10.0.0.73")).asString();
    user = bDBProperties.check("user", Value("postgres")).asString();
    password = bDBProperties.check("password", Value("icub")).asString();
    dataB = bDBProperties.check("dataB", Value("ABM")).asString();

    try {
        ABMDataBase = new DataBase<PostgreSql>(server, user, password, dataB);
        cout << "Database connected" << endl;
    }
    catch (DataBaseError e) {
        yError() << "Could not connect to database. Reason: " << e.what();
        return false;
    }

    if( rf.check("empty") == true )
        emptyTables();

    prepareDataBase();
    getPersonsIDs();

    return true;
}


void abmActions::prepareDataBase()
{
    *ABMDataBase << "CREATE TABLE IF NOT EXISTS persons (instance integer NOT NULL, opcid integer NOT NULL, name text NOT NULL, CONSTRAINT persons_pkey PRIMARY KEY (instance, opcid), FOREIGN KEY (instance, opcid) REFERENCES contentopc (instance, opcid));";
    *ABMDataBase << "ALTER TABLE persons OWNER TO postgres;";

}

void abmActions::getPersonsIDs()
{
    ostringstream osArg;

    personsOPCID.resize(personsNames.size());



    cout << "============ READING INSTANCE ID ==============" << endl;
    osArg.str("");
    osArg << "SELECT instance FROM main ORDER BY instance DESC LIMIT 1;";
    yInfo() << osArg.str();
    Bottle bRequest;
    Bottle response;
    bRequest.addString("request");
    bRequest.addString(osArg.str());
    response = iCub->getABMClient()->rpcCommand(bRequest);
    yInfo() << " Reponse de ABM : \n" << response.toString();

    int initInstanceValue = 0;
    if( !boost::iequals(response.toString(), "NULL") )
        initInstanceValue = boost::lexical_cast<int>(response.get(0).asList()->get(0).asString());

    if( initInstanceValue == 0 )
        instance = 0;
    else
        instance = initInstanceValue;

    cout << "Current instance value: " << instance << endl;


    cout << "============ READING OPCIDs ==============" << endl;
    for( int i = 0; i < personsNames.size(); i++ )
    {
        osArg.str("");
        osArg << "SELECT opcid FROM persons WHERE name = '" << personsNames.get(i).asString() << "';";
        yInfo() << osArg.str();
        bRequest.clear();
        response.clear();
        bRequest.addString("request");
        bRequest.addString(osArg.str());
        response = iCub->getABMClient()->rpcCommand(bRequest);
        yInfo() << " Reponse de ABM : \n" << response.toString();

        if( !boost::iequals(response.toString(), "NULL") )
        {
            cout << "Storing OPCIDs for person: " << personsNames.get(i).asString() << endl;
            for( int j = 0; j < response.size(); j++ )
                personsOPCID.at(i).addInt(boost::lexical_cast<int>(response.get(j).asList()->get(0).asString()));
            
            int maxID = personsOPCID.at(i).get(0).asInt();
            for( int j = 1; j < personsOPCID.at(i).size(); j++ )
            {
                if( personsOPCID.at(i).get(j).asInt() > maxID )
                    maxID = personsOPCID.at(i).get(j).asInt();
            }
            currentPersonsOPCID.addInt(maxID);
        }
        else
        {
            currentPersonsOPCID.addInt(defaultPersonsOPCID.get(i).asInt());
        }
    }


    cout << "============ STORED OPCIDs ==============" << endl;
    for( int i = 0; i < personsNames.size(); i++ )
    {
        cout << "OPCIDs for person: " << personsNames.get(i).asString() << endl;
        for( int j = 0; j < personsOPCID.at(i).size(); j++ )
        {
            cout << personsOPCID.at(i).get(j).asInt() << endl;
        }
        cout << endl;
    }


    cout << "============ CURRENT OPCIDs ==============" << endl;
    for( int i = 0; i < personsNames.size(); i++ )
    {
        cout << "OPCIDs for person: " << personsNames.get(i).asString() << endl;
        cout << currentPersonsOPCID.get(i).asInt() << endl;
    }
}

bool abmActions::interruptModule()
{
    return true;
}

double abmActions::getPeriod()
{
    return 0.1;
}

bool abmActions::close()
{
    iCub->close();
    delete iCub;

    return true;
}

string abmActions::currentDateTime()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

    return buf;
}

void abmActions::sendDataToABM(Bottle v_data)
{
    ostringstream osArg;
    int temp_instance = instance + 1;
    int temp_personOPCID = 0;

    string v_person = v_data.get(1).asString();
    string v_action = v_data.get(2).asString();
    string v_bodyPart = v_data.get(3).asString();
    string v_bodySide = v_data.get(4).asString();
    string v_direction = v_data.get(5).asString();
    string v_object = v_data.get(6).asString();
    
    Bottle tempIDValues;
    
    tempIDValues = currentPersonsOPCID;
    
    currentPersonsOPCID.clear();
    
    for( int i = 0; i < personsNames.size(); i++ )
    {
        if( boost::iequals(personsNames.get(i).asString(), v_person ) )
        {            
            // assigns current opcid for the corresponding persons            
            temp_personOPCID = tempIDValues.get(i).asInt();	

            // if the actions are performed with the same person, then the opcid doesn't change
            // else if there a new person recognised or the person sends the voice command to start a new set of actions, the opcid is incremented
            if( newActionProcess == true )
            {
                temp_personOPCID = temp_personOPCID + 1;    // increments the opcid in 1 for a new set actions started
                currentPersonsOPCID.addInt(temp_personOPCID);                
    	        newActionProcess = false;
            }
            else
                currentPersonsOPCID.addInt(tempIDValues.get(i).asInt());
        }
        else
            currentPersonsOPCID.addInt(tempIDValues.get(i).asInt());
    }
    
    cout << "NEW NAMES OPCID" << endl;
    for(int i = 0; i < personsNames.size(); i++ )
        cout << "Name: " << personsNames.get(i).asString() << ", OPCID: " << currentPersonsOPCID.get(i).asInt() << endl;


    osArg.str("");
    osArg << "INSERT INTO main(time,activityname,activitytype,instance,begin) VALUES('" << currentDateTime() << "','" << v_action << "','action'," << temp_instance <<",'TRUE');";
    cout << osArg.str() << endl;
    *ABMDataBase << osArg.str();


    osArg.str("");
    osArg << "INSERT INTO contentopc(instance,opcid,type,subtype) VALUES(" << temp_instance << "," << temp_personOPCID << ",'entity','agent');";
    cout << osArg.str() << endl;
    *ABMDataBase << osArg.str();

    osArg.str("");
    osArg << "INSERT INTO contentarg(instance,argument,type,subtype,role) VALUES(" << temp_instance << ",'" << v_person << "','external','default','person');";
    cout << osArg.str() << endl;
    *ABMDataBase << osArg.str();

    osArg.str("");
    osArg << "INSERT INTO contentarg(instance,argument,type,subtype,role) VALUES(" << temp_instance << ",'" << v_bodySide << "','external','default','" << v_bodyPart << "');";
    cout << osArg.str() << endl;
    *ABMDataBase << osArg.str();

    osArg.str("");
    osArg << "INSERT INTO contentarg(instance,argument,type,subtype,role) VALUES(" << temp_instance << ",'" << v_person << " was " << v_action << " using the " << v_bodySide << " " << v_bodyPart << "','external','default','sentence');";
    cout << osArg.str() << endl;
    *ABMDataBase << osArg.str();

    osArg.str("");
    osArg << "INSERT INTO entity(instance,opcid,name) VALUES(" << temp_instance << "," << temp_personOPCID << ",'person');";
    cout << osArg.str() << endl;
    *ABMDataBase << osArg.str();

    osArg.str("");
    osArg << "INSERT INTO action(instance,opcid,name,object,direction,argument) VALUES(" << temp_instance << "," << temp_personOPCID << ",'" << v_action << "','" << v_object << "','" << v_direction << "','" << v_bodySide << " " << v_bodyPart << "');";
    cout << osArg.str() << endl;
    *ABMDataBase << osArg.str();

    osArg.str("");
    osArg << "INSERT INTO agent(instance,opcid,name,presence) VALUES(" << temp_instance << "," << temp_personOPCID << ",'" << v_person << "','TRUE');";
    cout << osArg.str() << endl;
    *ABMDataBase << osArg.str();

    osArg.str("");
    osArg << "INSERT INTO persons(instance,opcid,name) VALUES(" << temp_instance << "," << temp_personOPCID << ",'" << v_person << "');";
    cout << osArg.str() << endl;
    *ABMDataBase << osArg.str();

    instance = temp_instance;
}

bool abmActions::getDataFromABM(Bottle v_data)
{
    int temp_personID = -1;
    int temp_personOPCID = 0;
    string v_person = v_data.get(1).asString();
    
    cout << "V_PERSON: " << v_person << endl;

    for( int i = 0; i < personsNames.size(); i++ )
    {
        if( boost::iequals(personsNames.get(i).asString(),v_person) )
        {
            temp_personID = i;
            temp_personOPCID = currentPersonsOPCID.get(temp_personID).asInt();
        }
    }
    
    if( temp_personID >= 0 )
    {
        cout << "NAME: " << personsNames.get(temp_personID).asString() << ", OPCID: " << temp_personOPCID << endl;

        ostringstream osArg;
        
        osArg << "SELECT name, object, direction, argument FROM action WHERE opcid = " << temp_personOPCID << ";";
        Bottle bRequest;
        Bottle response;
        bRequest.addString("request");
        bRequest.addString(osArg.str());
        response = iCub->getABMClient()->rpcCommand(bRequest);
        yInfo() << " Reponse de ABM : \n" << response.toString();
    }
    else
    {
        cout << "Person " << v_person << " not found in the database, no actions found." << endl;
        return false;
    }
    
    return true;
    
        // Code to retrieve data from database
/*
        osArg << "SELECT instance FROM contentarg WHERE argument='Uriel';";
        Bottle bRequest;
        Bottle response;
        bRequest.addString("request");
        bRequest.addString(osArg.str());
        response = iCub->getABMClient()->rpcCommand(bRequest);
  
        yInfo() << " Reponse de ABM : \n" << response.toString();
        
        yInfo() << "Size: " << response.size();
        
        for( int i = 0; i < response.size(); i++ )
        {
            ostringstream osArg2;
            Bottle response2;
            osArg2.clear();
            osArg2 << "SELECT argument FROM contentarg WHERE instance = " << response.get(i).asList()->get(0).asString() << " AND role = 'action';";
            yInfo() << osArg2.str();

            bRequest.clear();                    
            bRequest.addString("request");
            bRequest.addString(osArg2.str());
            response2.clear();
            response2 = iCub->getABMClient()->rpcCommand(bRequest);
            for( int j = 0; j < response2.size(); j++ )
            {
                yInfo() << "Action: " << response2.get(j).toString();
            }
        }
*/
}

/*
* Reset the tables : timeknowledge, timedata, spatialknowledge, spatialdata in the DataBase
*/
Bottle abmActions::emptyTables()
{
    Bottle bOutput;

    /****************************** Main Table ******************************/
    *ABMDataBase << "DROP TABLE IF EXISTS main CASCADE;";
    *ABMDataBase << "CREATE TABLE main(idActivity serial NOT NULL, time timestamp without time zone NOT NULL,activityname text, activitytype text, instance integer NOT NULL UNIQUE, begin boolean NOT NULL,CONSTRAINT main_pkey PRIMARY KEY (time)) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE main OWNER TO postgres;";

    /**************************** contentopc Table **************************/
    *ABMDataBase << "DROP TABLE IF EXISTS contentopc CASCADE;";
    *ABMDataBase << "CREATE TABLE contentopc(  instance integer NOT NULL,  opcid integer,  type text,  subtype text,  UNIQUE (instance, opcid),  CONSTRAINT contentopc_pkey PRIMARY KEY (instance, opcid),  FOREIGN KEY (instance) REFERENCES main (instance)) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE contentopc OWNER TO postgres;";

    /**************************** beliefs Table **************************/
    *ABMDataBase << "DROP TABLE IF EXISTS beliefs CASCADE;";
    *ABMDataBase << "CREATE TABLE beliefs (  instance integer NOT NULL,  idagent integer NOT NULL,  subject text NOT NULL,  verb text NOT NULL,  \"object\" text NOT NULL,  \"time\" text,  place text,  manner text,  CONSTRAINT beliefs_key FOREIGN KEY (instance) REFERENCES main (instance) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE NO ACTION) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE beliefs OWNER TO postgres;";

    /**************************** contentarg Table **************************/
    *ABMDataBase << "DROP TABLE IF EXISTS contentarg CASCADE;";
    *ABMDataBase << "CREATE TABLE contentarg(  instance integer NOT NULL,  argument text, type text, subtype text, role text, UNIQUE (instance, role, argument),  CONSTRAINT contentarg_pkey PRIMARY KEY (instance, role, argument),  FOREIGN KEY (instance) REFERENCES main (instance)) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE contentarg OWNER TO postgres;";

    /****************************** entity Table ****************************/
    *ABMDataBase << "DROP TABLE IF EXISTS entity CASCADE;";
    *ABMDataBase << "CREATE TABLE entity(  instance int NOT NULL,  opcid integer NOT NULL,  name text NOT NULL,  CONSTRAINT entity_pkey PRIMARY KEY (instance, opcid),  UNIQUE (instance, opcid),  FOREIGN KEY (instance, opcid) REFERENCES contentopc (instance, opcid)) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE entity OWNER TO postgres;";

    /****************************** action Table ****************************/
    //herits from entity
    *ABMDataBase << "DROP TABLE IF EXISTS action CASCADE;";
    *ABMDataBase << "CREATE TABLE action(  instance int NOT NULL,  opcid integer NOT NULL,  name text NOT NULL, object text NOT NULL, direction text NOT NULL, argument text,  CONSTRAINT action_pkey PRIMARY KEY (instance, opcid),  UNIQUE (opcid, instance),  FOREIGN KEY (instance, opcid) REFERENCES contentopc (instance, opcid)) INHERITS (entity) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE action OWNER TO postgres;";

    /**************************** adjective Table **************************/
    //herits from entity
    *ABMDataBase << "DROP TABLE IF EXISTS adjective CASCADE;";
    *ABMDataBase << "CREATE TABLE adjective(  quality text,  CONSTRAINT adjective_pkey PRIMARY KEY (instance, opcid),  UNIQUE (opcid, instance),  FOREIGN KEY (instance, opcid) REFERENCES contentopc (instance, opcid)) INHERITS (entity) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE adjective OWNER TO postgres;";

    /****************************** object Table ***************************/
    //herits from entity
    *ABMDataBase << "DROP TABLE IF EXISTS object CASCADE;";
    *ABMDataBase << "CREATE TABLE object(  presence boolean NOT NULL,  position real [],  orientation real[],  dimension real[],  color int[], saliency real, CONSTRAINT object_pkey PRIMARY KEY (instance, opcid),  UNIQUE (instance, opcid),  FOREIGN KEY (instance, opcid) REFERENCES contentopc (instance, opcid)) INHERITS (entity) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE object OWNER TO postgres;";

    /****************************** rtobject Table ***************************/
    //herits from object
    *ABMDataBase << "DROP TABLE IF EXISTS rtobject CASCADE;";
    *ABMDataBase << "CREATE TABLE rtobject(  rtposition real[],  CONSTRAINT rtobject_pkey PRIMARY KEY (instance, opcid),  UNIQUE (opcid, instance),  FOREIGN KEY (instance, opcid) REFERENCES contentopc (instance, opcid)) INHERITS (object) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE rtobject OWNER TO postgres;";

    /****************************** agent Table ***************************/
    //herits from entity
    *ABMDataBase << "DROP TABLE IF EXISTS agent CASCADE;";
    *ABMDataBase << "CREATE TABLE agent(  emotion text[],  CONSTRAINT agent_pkey PRIMARY KEY (instance, opcid),  UNIQUE (opcid, instance),  FOREIGN KEY (instance, opcid) REFERENCES contentopc (instance, opcid)) INHERITS (object) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE agent OWNER TO postgres;";

    /****************************** relation Table ***************************/
    *ABMDataBase << "DROP TABLE IF EXISTS relation CASCADE;";
    *ABMDataBase << "CREATE TABLE relation(  instance integer NOT NULL,  opcid integer NOT NULL,  subject text NOT NULL,  verb text NOT NULL, object text, time text,  place text,  manner text,  CONSTRAINT relation_pkey PRIMARY KEY (instance, opcid),  UNIQUE (instance,opcid),  FOREIGN KEY (instance, opcid) REFERENCES contentopc (instance, opcid) ) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE relation OWNER TO postgres;";

    /***************************** Drives table *****************************/
    *ABMDataBase << "DROP TABLE IF EXISTS drives CASCADE;";
    *ABMDataBase << "CREATE TABLE drives (     instance integer NOT NULL,  name     text NOT NULL, value double precision NOT NULL, homeomax double precision NOT NULL, homeomin double precision NOT NULL, UNIQUE (instance, name ) ,    CONSTRAINT drives_pkey     PRIMARY KEY (instance, name),     FOREIGN KEY (instance) REFERENCES main (instance),  UNIQUE (name, instance) ) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE drives OWNER TO postgres;";

    /***************************** Emotions table *****************************/
    *ABMDataBase << "DROP TABLE IF EXISTS emotions CASCADE;";
    *ABMDataBase << "CREATE TABLE emotions ( instance integer NOT NULL,  name text NOT NULL, value double precision NOT NULL, UNIQUE (name, instance ) ,    CONSTRAINT emotion_pkey     PRIMARY KEY (instance, name),     FOREIGN KEY (instance) REFERENCES main (instance),  UNIQUE (instance, name) ) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE drives OWNER TO postgres;";

    /****************************** spatialknowledge *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS spatialknowledge CASCADE;";
    *ABMDataBase << "CREATE TABLE spatialknowledge ( name text NOT NULL,  argument text NOT NULL, dependance text NOT NULL , instance integer NOT NULL,  CONSTRAINT spatialknowledge_pkey PRIMARY KEY (instance),  CONSTRAINT spatialknowledge_name_key UNIQUE (name, argument, dependance) ) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE spatialknowledge OWNER TO postgres;";

    /****************************** spatialdata *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS spatialdata CASCADE;";
    *ABMDataBase << "CREATE TABLE spatialdata ( vx double precision, vy double precision, vdx double precision, vdy double precision, instance integer NOT NULL, id serial NOT NULL, CONSTRAINT spatialdata_instance_fkey FOREIGN KEY (instance) REFERENCES spatialknowledge (instance) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE NO ACTION ) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE spatialdata OWNER  TO postgres;";

    /****************************** contextknowledge *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS contextknowledge CASCADE;";
    *ABMDataBase << "CREATE TABLE contextknowledge ( name text NOT NULL,  argument text NOT NULL,  dependance text NOT NULL, instance integer NOT NULL,  CONSTRAINT contextknowledge_pkey PRIMARY KEY (instance),  CONSTRAINT contextknowledge_name_key UNIQUE (name, argument, dependance) ) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE contextknowledge OWNER TO postgres;";

    /****************************** contextdata *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS contextdata CASCADE;";
    *ABMDataBase << "CREATE TABLE contextdata (  presencebegin boolean,  presenceend boolean,  instance integer NOT NULL,  id serial NOT NULL,  CONSTRAINT contextdata_instance_fkey FOREIGN KEY (instance) REFERENCES contextknowledge (instance) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE NO ACTION ) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE contextdata OWNER  TO postgres;";

    /****************************** contextagent *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS contextagent CASCADE;";
    *ABMDataBase << "CREATE TABLE contextagent ( instance integer NOT NULL, agent text NOT NULL, number integer, CONSTRAINT contextagent_pkey PRIMARY KEY (instance, agent), CONSTRAINT contextagent_instance_fkey FOREIGN KEY (instance) REFERENCES contextknowledge (instance) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE NO ACTION) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE contextagent OWNER  TO postgres;";

    /****************************** timeknowledge *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS timeknowledge CASCADE;";
    *ABMDataBase << "CREATE TABLE timeknowledge ( temporal text NOT NULL,   CONSTRAINT timeknowledge_pkey PRIMARY KEY (temporal) ) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE timeknowledge OWNER TO postgres;";

    /****************************** timedata *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS timedata CASCADE;";
    *ABMDataBase << "CREATE TABLE timedata ( temporal text NOT NULL,   timearg1 timestamp without time zone NOT NULL,   timearg2 timestamp without time zone NOT NULL,   CONSTRAINT timedata_temporal_fkey FOREIGN KEY (temporal)        REFERENCES timeknowledge (temporal) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE NO ACTION ) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE timedata OWNER  TO postgres;";

    /****************************** interactionknowledge *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS interactionknowledge CASCADE;";
    *ABMDataBase << "CREATE TABLE interactionknowledge (subject text NOT NULL, argument text NOT NULL, number integer NOT NULL, type text NOT NULL DEFAULT 'none'::text, role text NOT NULL DEFAULT 'none'::text, CONSTRAINT interactionknowledge_pkey PRIMARY KEY (subject, argument, type, role) ) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE interactionknowledge OWNER  TO postgres;";

    /****************************** behavior *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS behavior CASCADE;";
    *ABMDataBase << "    CREATE TABLE behavior(  \"name\" text NOT NULL,  argument text NOT NULL,  instance integer NOT NULL,  CONSTRAINT behavior_pkey PRIMARY KEY (instance),  CONSTRAINT behavior_name_key UNIQUE (name, argument)) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE behavior OWNER  TO postgres;";

    /****************************** behaviordata *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS behaviordata CASCADE;";
    *ABMDataBase << "    CREATE TABLE behaviordata(  drive text NOT NULL,  effect double precision,  instance integer NOT NULL,  occurence integer NOT NULL,  CONSTRAINT behaviordata_pkey PRIMARY KEY (occurence, instance, drive),  CONSTRAINT behaviordata_instance_fkey FOREIGN KEY (instance)      REFERENCES behavior (instance) MATCH SIMPLE      ON UPDATE NO ACTION ON DELETE NO ACTION)WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE behaviordata OWNER  TO postgres;";

    /****************************** sharedplan *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS sharedplan CASCADE;";
    *ABMDataBase << "       CREATE TABLE sharedplan(  \"name\" text NOT NULL,  manner text NOT NULL,  instance integer NOT NULL,  CONSTRAINT sharedplan_pkey PRIMARY KEY (instance),  CONSTRAINT sharedplan_name_key UNIQUE (name, manner))WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE sharedplan OWNER  TO postgres;";

    /****************************** sharedplanarg *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS sharedplanarg CASCADE;";
    *ABMDataBase << "     CREATE TABLE sharedplanarg(  instance integer NOT NULL,  argument text NOT NULL,  \"role\" text NOT NULL,  CONSTRAINT sharedplanarg_pkey PRIMARY KEY (instance, role, argument),  CONSTRAINT sharedplanarg_instance_fkey FOREIGN KEY (instance)      REFERENCES sharedplan (instance) MATCH SIMPLE      ON UPDATE NO ACTION ON DELETE NO ACTION)WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE sharedplanarg OWNER  TO postgres;";

    /****************************** sharedplandata *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS sharedplandata CASCADE;";
    *ABMDataBase << "     CREATE TABLE sharedplandata (  activitytype text NOT NULL,  activityname text NOT NULL,  instance integer NOT NULL,  id integer NOT NULL,  CONSTRAINT sharedplandata_pkey PRIMARY KEY (instance, id),  CONSTRAINT sharedplandata_instance_fkey FOREIGN KEY (instance)      REFERENCES sharedplan (instance) MATCH SIMPLE      ON UPDATE NO ACTION ON DELETE NO ACTION)WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE sharedplandata OWNER  TO postgres;";

    /****************************** spdataarg *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS spdataarg CASCADE;";
    *ABMDataBase << "CREATE TABLE spdataarg (id integer NOT NULL, instance integer NOT NULL, argument text NOT NULL, \"role\" text NOT NULL, CONSTRAINT spdataarg_pkey PRIMARY KEY (id, instance, role, argument), CONSTRAINT spdataarg_instance_fkey FOREIGN KEY (instance, id) REFERENCES sharedplandata (instance, id) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE NO ACTION) WITH (OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE spdataarg OWNER  TO postgres;";

    /****************************** visualdata *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS visualdata CASCADE;";
    *ABMDataBase << "CREATE TABLE visualdata(\"time\" timestamp without time zone NOT NULL, img_provider_port text NOT NULL, instance integer NOT NULL, frame_number integer NOT NULL, relative_path text NOT NULL, augmented text, img_oid oid, CONSTRAINT img_pkey PRIMARY KEY(\"time\", img_provider_port), CONSTRAINT visualdata_instance_fkey FOREIGN KEY(instance) REFERENCES main(instance) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE NO ACTION) WITH(OIDS = FALSE);";
    *ABMDataBase << "ALTER TABLE visualdata OWNER TO postgres;";
    *ABMDataBase << "CREATE INDEX visualdata_instance_time ON visualdata (instance, time);";

    /****************************** proprioceptivedata *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS proprioceptivedata CASCADE;";
    *ABMDataBase << "CREATE TABLE proprioceptivedata(instance integer NOT NULL, \"time\" timestamp without time zone NOT NULL, label_port text NOT NULL, subtype text NOT NULL, frame_number integer NOT NULL, value text NOT NULL, CONSTRAINT cont_pkey PRIMARY KEY (\"time\", label_port, subtype), CONSTRAINT proprio_instance_fkey FOREIGN KEY (instance) REFERENCES main (instance) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE NO ACTION) WITH ( OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE proprioceptivedata OWNER TO postgres;";
    *ABMDataBase << "CREATE INDEX proprioceptivedata_instance_time ON proprioceptivedata (instance, time);";

    /****************************** adjectivespatial *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS adjectivespatial CASCADE;";
    *ABMDataBase << "CREATE TABLE adjectivespatial ( \"name\" text NOT NULL, argument text NOT NULL, x double precision, y double precision, dx double precision, dy double precision ) WITH(OIDS = FALSE) ";
    *ABMDataBase << "ALTER TABLE adjectivespatial OWNER TO postgres;";

    /****************************** adjectivetemporal *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS adjectivetemporal CASCADE;";
    *ABMDataBase << "CREATE TABLE adjectivetemporal ( \"name\" text NOT NULL, argument text NOT NULL, timing double precision) WITH(OIDS = FALSE) ";
    *ABMDataBase << "ALTER TABLE adjectivetemporal OWNER TO postgres;";

    /****************************** sounddata *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS sounddata CASCADE;";
    *ABMDataBase << "CREATE TABLE sounddata(\"time\" timestamp without time zone NOT NULL, snd_provider_port text NOT NULL, instance integer NOT NULL, relative_path text, snd_oid oid, CONSTRAINT snd_pkey PRIMARY KEY (\"time\", snd_provider_port), CONSTRAINT sound_instance_fkey FOREIGN KEY (instance) REFERENCES main (instance) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE NO ACTION ) WITH (  OIDS=FALSE);";
    *ABMDataBase << "ALTER TABLE sounddata OWNER TO postgres;";

    /****************************** sentencedata *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS sentencedata CASCADE;";
    *ABMDataBase << "CREATE TABLE sentencedata ( instance integer NOT NULL, word text, \"role\" text, \"level\" integer NOT NULL, CONSTRAINT sentencedata_pkey PRIMARY KEY(instance, level), CONSTRAINT sentencedata_instance_fkey FOREIGN KEY(instance) REFERENCES main(instance) MATCH SIMPLE ON UPDATE NO ACTION ON DELETE NO ACTION ) WITH(OIDS = FALSE);";
    *ABMDataBase << "ALTER TABLE sentencedata OWNER TO postgres;";

    /****************************** personsdata *************************/
    *ABMDataBase << "DROP TABLE IF EXISTS persons CASCADE;";
    *ABMDataBase << "CREATE TABLE persons (instance integer NOT NULL, opcid integer NOT NULL, name text NOT NULL, CONSTRAINT persons_pkey PRIMARY KEY (instance, opcid), FOREIGN KEY (instance, opcid) REFERENCES contentopc (instance, opcid));";
    *ABMDataBase << "ALTER TABLE persons OWNER TO postgres;";


    bOutput.addString("knowledge database reset");
    return bOutput;
}

