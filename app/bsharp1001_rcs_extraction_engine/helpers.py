
from collections import defaultdict
from collections import defaultdict
import uuid
import regex
import json
import shortuuid

SEGSEP='#____SEP____#'

def rec_dd():
    return defaultdict(rec_dd)

class extractedForm:
    def __init__(self, form_id, rcs_number=None, filing_id=None, filing_date=None, title = "", subtitle = "", page_count = None, existingSections = [], absentSections = [], extracted_sections = {}):
        self.rcs_number = rcs_number
        self.filing_id = filing_id
        self.filing_date = filing_date
        self.form_id = form_id
        self.title = title
        self.subtitle = subtitle
        self.page_count = page_count
        self.existingSections = existingSections
        self.absentSections = absentSections
        self.extracted_sections = extracted_sections
        self.sdprecordsstandard = rec_dd()
        self.sdprecords = rec_dd()
        self.ownerrecords: list[ListEntity] = []
        # Load mapping files from current working directory
        # The service sets working directory to mappings folder
        import os
        # Try current directory first (when running as microservice)
        mappings_dir = os.getcwd()
        data_mapping_path = os.path.join(mappings_dir, "extracted_data_mapping.json")
        section_mapping_path = os.path.join(mappings_dir, "section_mapping.json")
        table_dp_mapping_path = os.path.join(mappings_dir, "table_dp_mapping.json")
        
        # Fallback to relative paths if files not found in current directory
        if not os.path.exists(data_mapping_path):
            data_mapping_path = "./extracted_data_mapping.json"
        if not os.path.exists(section_mapping_path):
            section_mapping_path = "./section_mapping.json"
        if not os.path.exists(table_dp_mapping_path):
            table_dp_mapping_path = "./table_dp_mapping.json"
        
        self.data_mapping = json.loads(open(data_mapping_path, 'r', encoding='utf-8').read())
        self.table_mapping = json.loads(open(section_mapping_path, 'r', encoding='utf-8').read())
        self.table_dp_mapping = json.loads(open(table_dp_mapping_path, 'r', encoding='utf-8').read())
        self.paths_to_delete = []
        self.leftovers = {}
        self.rcs_dps = ["DP_520", "DP_060", "DP_062", "DP_071", "DP_082", "DP_074", "DP_568", "DP_590", "DP_614", "DP_662", "DP_702"]
    
    def serializeSelf(self):
        return {
            "rcs_number": self.rcs_number,
            "filing_id": self.filing_id,
            "filing_date": self.filing_date,
            "form_id": self.form_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "page_count": self.page_count,
            "existingSections": self.existingSections,
            "absentSections": self.absentSections,
            "extracted_sections": self.extracted_sections,
            "sdprecords": self.sdprecords,
            "sdprecordsstandard": self.sdprecordsstandard,
            "leftovers": self.leftovers,
            "table_dp_mapping": self.table_dp_mapping,
            }
    
    def serializeToJSON(self):
        dict = self.serializeSelf()
        return json.dumps(dict)
    
    def replacePredefinedSuffixes(self, s: str):
        return s.split(SEGSEP)[-1].replace("subtitles_list", "").replace("data_list", "").replace("_subtitle", "")
    
    def normalizeString(self, word: str):
        word = word.lower().replace(":", "").strip()
        word = regex.regex.sub(r'\s{2,}', ' ', word.replace("(s)", "").replace("(er)", "").replace("(es)", "").replace("(e)", "").replace("(en)", ""), flags= regex.MULTILINE | regex.IGNORECASE)
        word = regex.regex.sub(r'(?<=\s)\w(?=\s)', r'###\g<0>###', word, flags= regex.MULTILINE | regex.IGNORECASE).strip()
        word = regex.regex.sub(r'\s*###\s*', '', word, flags= regex.MULTILINE | regex.IGNORECASE)
        word = regex.regex.sub(r'(?<=\(Art\.)\s*.*\s*(?=LSC\))', ' ', word, flags= regex.MULTILINE | regex.IGNORECASE)
        return word
    
    def extractedExistsinDefined(self, sections: list[str], extracted_section: str):
        #print(f"extractedExistsinDefined: {extracted_section}")
        if sections == "all":
            return True
        for defined in sections:
            d = self.normalizeString(defined)
            f = self.normalizeString(extracted_section)
            if regex.match(r'^(nou\w*|neu\w*|\W*-)?\s*'+ regex.escape(f) +r'\s*$', d):
                return True
        return False
    
    def getSectionTable(self, sections: dict, extracted_section: str):
        #print(f"getSectionTable: {extracted_section}")
        for key in sections.keys():
            for s in sections[key]:
                d = self.normalizeString(s)
                f = self.normalizeString(extracted_section).lower().strip()
                if regex.match(r'^(nou\w*|neu\w*|\W*-)?\s*'+ regex.escape(f) +r'\s*$', d):
                    return key
        return "no_corresponding_table"
    
    def fieldExists(self, fields: list[str], extracted_field: str):
        for s in fields:
            if self.normalizeString(s).replace(":", "").lower().strip() == self.normalizeString(extracted_field).replace(":", "").lower().strip():
                return True
        return False
    
    def getTableDPs(self, base_table):
        base_table = regex.regex.sub(r'(?<=_\d{1,})_[A-Z]{1,}$', '', base_table)
        ll = []
        [ ll.extend(self.table_dp_mapping[t]) for t in self.table_dp_mapping.keys() if t.startswith(base_table)]
        return ll

    def restructureDataforDB(self, breadcrumbs: list[str], dict_ = None,):
        #if dict_ is None:
        #    self.esc = self.extracted_sections
        target = dict_ if dict_ is not None else self.extracted_sections
        targetKeys = target.keys()
        for key in targetKeys:
            breadcrumbs.append(key)
            if key == "headers":
                self.paths_to_delete.append(breadcrumbs.copy())
                breadcrumbs.pop()
                continue
            if type(target[key]) is dict:
                self.restructureDataforDB(breadcrumbs=breadcrumbs, dict_= target[key])
            elif type(target[key]) is list:
                for entity in target[key]:
                    entity_sepd = entity[1].split(SEGSEP)
                    table = self.getSectionTable(self.table_mapping, entity_sepd[0])
                    self.ownerrecords.append(ListEntity(type_=entity_sepd[0], name=entity_sepd[-1], table=table, hash=shortuuid.ShortUUID().random(length=28), form_id=self.form_id))
                self.paths_to_delete.append(breadcrumbs.copy())
            else:
                parent_section = None
                owner_id = None
                owner_table = None
                owner_hash = None
                owner_type = None
                owner_name = None
                list_entry_id = None
                data_mapping = None
                item = self.replacePredefinedSuffixes(breadcrumbs[-2])
                isItem = regex.match(r'item[0-9]{1,4}', item)
                if isItem is not None:
                    list_entry_id = int(self.replacePredefinedSuffixes(breadcrumbs[-2]).replace("item", ''))
                    bcwithoutSuffixes = [self.replacePredefinedSuffixes(d) for d in breadcrumbs if self.replacePredefinedSuffixes(d) != ""]
                    parent_section = self.replacePredefinedSuffixes(bcwithoutSuffixes[-3])
                else:
                    parent_section = self.replacePredefinedSuffixes(breadcrumbs[-2])
                
                field_name_in_form = self.replacePredefinedSuffixes(key)
                base_table = self.getSectionTable(self.table_mapping, breadcrumbs[0].split(SEGSEP)[0])
                base_table = regex.regex.sub(r'(?<=_\d{1,})_\d{1,}$', '', base_table)
                if breadcrumbs[0].__contains__(SEGSEP):
                    [owner_type, owner_name] = [breadcrumbs[0].split(SEGSEP)[0], breadcrumbs[0].split(SEGSEP)[-1]]
                    for id, owner in enumerate(self.ownerrecords):
                        if owner.type == owner_type and owner.name == owner_name:
                            owner_id = id
                            owner_table = owner.table
                            owner_hash = owner.hash
                            break 
                    for mapping in self.data_mapping:
                        if self.fieldExists(mapping['field_name'], field_name_in_form) and (self.extractedExistsinDefined(mapping['section'], parent_section) and mapping['dp_id'] in self.getTableDPs(base_table)):
                            data_mapping = mapping
                            break
                        
                        elif self.fieldExists(mapping['field_name'], field_name_in_form) and self.extractedExistsinDefined(mapping['section'], owner_type):
                            data_mapping = mapping
                            break
                else:
                    for mapping in self.data_mapping:
                        if self.fieldExists(mapping['field_name'], field_name_in_form) and self.extractedExistsinDefined(mapping['section'], parent_section):
                            data_mapping = mapping
                            break
                                
                if data_mapping is not None:
                    table = None
                    #dps on more than one table
                    if data_mapping['dp_id'] in ['DP_503', 'DP_569', 'DP_680', "DP_524", "DP_527", "DP_528", "DP_526", "DP_536", "DP_537", "DP_540", "DP_541", "DP_542", "DP_529", "DP_533", "DP_530", "DP_534", "DP_531", "DP_532", "DP_535", "DP_564"]:
                        table = self.getSectionTable(self.table_mapping, owner_type) if owner_type is not None else self.getSectionTable(self.table_mapping, breadcrumbs[0].split(SEGSEP)[0])
                    else:
                        for tdpm in self.table_dp_mapping.keys():
                            if data_mapping['dp_id'] in self.table_dp_mapping[tdpm]:
                                table = tdpm
                                break
                    
                    #mmm = table is not None and regex.regex.search(r'_\d+$', table.strip()) != None
                    #jjj = base_table is not None and regex.regex.search(r'_\d+$', base_table.strip()) != None
                    #ddd = table is not None and table.startswith(base_table) and mmm and jjj
                    #if ddd == False:
                    #    breadcrumbs.pop()
                    #    continue
                    
                    owner_id_std = owner_id
                    ouuid = False
                    if table is not None and owner_id is not None and owner_table is not None and table != owner_table:
                        if regex.regex.search(r'(?<=_\d{1,})_\d{1,}$', table) != None and regex.regex.search(r'(?<=_\d{1,})_\d{1,}$', owner_table) != None:
                            owner_id_std = owner_table+"_standard"+"##"+str(owner_hash)
                            owner_id = owner_table+"##"+str(owner_hash)
                    
                        elif regex.regex.search(r'(?<=_\d{1,})_\d{1,}$', table) == None and regex.regex.search(r'(?<=_\d{1,})_\d{1,}$', owner_table) == None:
                            owner_id_std = None
                            owner_id = None
                    
                    if table is not None and owner_id is not None and owner_table is not None and table == owner_table:
                        ouuid = owner_hash
                        
                    if table is not None and owner_id is not None and list_entry_id is not None:
                        self.sdprecords[table][owner_id][list_entry_id][data_mapping['dp_id']] = target[key]
                        self.sdprecordsstandard[table+"_standard"][owner_id_std][list_entry_id][data_mapping['standard_name']] = target[key]
                        
                        #self.sdprecords[table][owner_id][list_entry_id]['rcs_number'] = self.rcs_number 
                        #self.sdprecords[table][owner_id][list_entry_id]['filing_id'] = self.filing_id 
                        #self.sdprecords[table][owner_id][list_entry_id]['file_hash'] = self.form_id 
                        #self.sdprecordsstandard[table+"_standard"][owner_id_std][list_entry_id]['file_hash'] = self.form_id 
                        
                        if ouuid:
                            self.sdprecords[table][owner_id][list_entry_id]['uuid'] = ouuid
                            self.sdprecordsstandard[table+"_standard"][owner_id_std][list_entry_id]['uuid'] = ouuid

                        self.paths_to_delete.append(breadcrumbs.copy())
                    elif table is not None and owner_id is not None:
                        self.sdprecords[table][owner_id][data_mapping['dp_id']] = target[key]
                        self.sdprecordsstandard[table+"_standard"][owner_id_std][data_mapping['standard_name']] = target[key]
                        
                        #self.sdprecords[table][owner_id]['rcs_number'] = self.rcs_number 
                        #self.sdprecords[table][owner_id]['filing_id'] = self.filing_id 
                        #self.sdprecords[table][owner_id]['file_hash'] = self.form_id 
                        #self.sdprecordsstandard[table+"_standard"][owner_id_std]['file_hash'] = self.form_id 
                        
                        if ouuid:
                            tap = regex.sub(r".*:\s*", '', owner_name)
                            tap = regex.regex.sub(r'\s+(?=\w\s+)', '', tap).strip()
                            newlinep = regex.compile(r".*"+ r'(' + '|'.join([ e.replace(' ', r'\s*') for e in self.table_mapping[table]]) + r')' +r"\s*", flags=regex.IGNORECASE | regex.MULTILINE)
                            tap = regex.sub(newlinep, '', tap)
                            for k in self.sdprecords[table][owner_id].keys():
                                if k in self.rcs_dps and not tap.startswith(self.sdprecords[table][owner_id][k]):
                                    tap = self.sdprecords[table][owner_id][k] + " - " +  tap
                            self.sdprecords[table][owner_id]['title_as_appeared'] = tap
                            self.sdprecords[table][owner_id]['uuid'] = ouuid
                            self.sdprecordsstandard[table+"_standard"][owner_id_std]['uuid'] = ouuid
                        
                        self.paths_to_delete.append(breadcrumbs.copy())
                    elif table is not None:
                        self.sdprecords[table][data_mapping['dp_id']] = target[key]
                        self.sdprecordsstandard[table+"_standard"][data_mapping['standard_name']] = target[key],
                        
                        #self.sdprecords[table]['rcs_number'] = self.rcs_number 
                        #self.sdprecords[table]['filing_id'] = self.filing_id 
                        #self.sdprecords[table]['file_hash'] = self.form_id
                        #self.sdprecordsstandard[table+"_standard"]['file_hash'] = self.form_id 
                        
                        if ouuid:
                            self.sdprecords[table]['uuid'] = ouuid
                            self.sdprecordsstandard[table+"_standard"]['uuid'] = ouuid
                            
                        self.paths_to_delete.append(breadcrumbs.copy())
            breadcrumbs.pop()
        return
    
    def postprocessSections(self):
        return
    
    def preprocessSections(self):
        return
    
    def remove_empty(self, dict_ = None, keyMap = None):
        dic = dict_ if dict_ != None else self.extracted_sections
        kk = keyMap if keyMap != None else self.paths_to_delete
        for keys in kk:
            for key in keys:
                if dic.keys().__contains__(key) == False:
                    break
                if type(dic[key]) is dict and len(dic[key]) > 0:
                    fd = keys.copy()
                    fd.pop(0)
                    self.remove_empty(dict_=dic[key], keyMap=[fd])
                if type(dic[key]) is dict and len(dic[key]) == 0:
                    del dic[key]
                    break
    
    def nested_del(self):
        for keys in self.paths_to_delete:
            dic = self.extracted_sections
            for key in keys[:-1]:
                dic = dic[key]
            del dic[keys[-1]]
    
    def nested_rename(self, dd= None):
        hh = {}
        dic = dd if dd is not None else self.extracted_sections
        for key in dic.keys():
            keyl = key
            if key.__contains__(SEGSEP):
                table = self.getSectionTable(self.table_mapping, key.split(SEGSEP)[0])
                keyl = table+"_"+key.split(SEGSEP)[0]+"_"+str(shortuuid.ShortUUID().random(length=5))+"_"+key.split(SEGSEP)[-1]
            if type(dic[key]) is dict:
                hh[keyl] = self.nested_rename(dic[key])
            else:
                hh[keyl] = dic[key]
        return hh

class T3ExtractedForm(extractedForm):
    def postprocessSections(self):
        sections = self.sdprecords.keys()
        for section in sections:
            dps_joined = "::".join(self.table_dp_mapping[section])
            dps = self.table_dp_mapping[section]
            dp_objects = [dp for dp in self.data_mapping if dp["dp_id"] in dps]
            reps = defaultdict(rec_dd)
            if dps_joined.__contains__("RDP"):
                section_keys = self.sdprecords[section].keys()
                section_count = len(section_keys) + 1
                for nkey in section_keys:
                    rdps_to_remove = []
                    increase_count = False
                    for dp_key in self.sdprecords[section][nkey].keys():
                        if dp_key.startswith("RDP"):
                            rdop_standard_name = [dp["standard_name"] for dp in dp_objects if dp["dp_id"] == dp_key][0]
                            matching_dp = [dp["dp_id"] for dp in dp_objects if dp['standard_name'].endswith(rdop_standard_name) and dp["dp_id"] != dp_key][0]
                            reps[section_count][matching_dp] = self.sdprecords[section][nkey][dp_key]
                            rdps_to_remove.append(dp_key)
                            increase_count = True
                    if increase_count:
                        reps[section_count]["is_representative"] = True
                        reps[section_count]["uuid"] = uuid.uuid4()
                        reps[section_count]["title_as_appeared"] = f"{self.sdprecords[section][nkey]['RDP_011']} {self.sdprecords[section][nkey]['RDP_012']}"
                        self.sdprecords[section][nkey]["representative_id"] = reps[section_count]["uuid"]
                        section_count = section_count + 1
                    for r in rdps_to_remove:
                        del self.sdprecords[section][nkey][r]
            if len(reps) > 0:
                final = defaultdict(rec_dd)
                final.update({**self.sdprecords[section], **reps})
                self.sdprecords[section] = final
    
    def preprocessSections(self):
        for key in self.extracted_sections.keys():
            section_keys = self.extracted_sections[key].keys()
            index = -1
            repkey = ""
            keys_to_remove = []
            for x, k in enumerate(section_keys):
                c1 = self.extractedExistsinDefined(["Représentant permanent", "Représentant permanent de la personne morale"], k.split(SEGSEP)[-1].replace("_subtitle", ""))
                if index != -1 and x > index:
                    keys_to_remove.append(k)
                    self.extracted_sections[key][repkey] = {**self.extracted_sections[key][repkey], **self.extracted_sections[key][k]}
                elif k.__contains__(SEGSEP) and c1:
                    index = x
                    repkey = k
            for r in keys_to_remove:
                del self.extracted_sections[key][r]
    

class T4ExtractedForm(extractedForm):
    def __init__(self, form_id, rcs_number=None, filing_id=None, filing_date=None, title = "", subtitle = "", page_count = None, existingSections = [], absentSections = [], extracted_sections = {}):
        super().__init__(form_id, rcs_number, filing_id, filing_date, title, subtitle, page_count, existingSections, absentSections, extracted_sections)
        self.statements_mapping = json.loads(open("./statements_mapping.json", 'r').read())
        self.tables_to_map_section_names_to_dps = {"ta_013": "DP_063"}
    
    def map_statement(self, statement):
        for key in self.statements_mapping.keys():
            if statement == key:
                return self.statements_mapping[key]['dps']
        return {}
    
    def strNullOrEmpty(self, string):
        return string == None or str(string).strip() == ""
        
    def postprocessSections(self):
        sections = self.sdprecords.keys()
        
        for section in sections:
            if str(section).lower() == "ta_006" and self.sdprecords[section].keys().__contains__("DP_053"):
                dp_053 = str(self.sdprecords[section]["DP_053"]).strip()
                if (
                    self.strNullOrEmpty(dp_053) == False
                    and 
                    (
                        self.sdprecords[section].keys().__contains__("DP_051") == False
                        or
                        (self.sdprecords[section].keys().__contains__("DP_051") and self.strNullOrEmpty(str(self.sdprecords[section]["DP_051"]).strip()))
                    )
                ):
                    self.sdprecords[section]["DP_051"] = "Fixe"
            
            elif str(section).lower() == "ta_008" and self.sdprecords[section].keys().__contains__("DP_005"):
                dp_005 = str(self.sdprecords[section]["DP_005"]).strip()
                if (
                    self.strNullOrEmpty(dp_005) == False
                    and 
                    (
                        self.sdprecords[section].keys().__contains__("DP_002") == False
                        or
                        (self.sdprecords[section].keys().__contains__("DP_002") and self.strNullOrEmpty(str(self.sdprecords[section]["DP_002"]).strip()))
                    )
                ):
                    self.sdprecords[section]["DP_002"] = "Déterminée"
            
            else:
                dps_joined = "::".join(self.table_dp_mapping[section])
                dps = self.table_dp_mapping[section]
                dp_objects = [dp for dp in self.data_mapping if dp["dp_id"] in dps]
                reps = defaultdict(rec_dd)
                if dps_joined.__contains__("RDP"):
                    rep_rdp = [dp for dp in dp_objects if dp['standard_name'].endswith("PermanentRepresentative")][0]
                    rdps = [dp for dp in dp_objects if dp['dp_id'].startswith("RDP")]
                    section_keys = self.sdprecords[section].keys()
                    section_count = len(section_keys) + 1
                    for nkey in section_keys:
                        rdps_to_remove = []
                        increase_count = False
                        person_keys = self.sdprecords[section][nkey].keys()
                        #for dp_key in :
                        if rep_rdp["dp_id"] in person_keys:
                            for dp_key in rdps:
                                rdp_key = dp_key["dp_id"]
                                rdop_standard_name = [dp["standard_name"] for dp in dp_objects if dp["dp_id"] == rdp_key][0]
                                matching_dp = [dp["dp_id"] for dp in dp_objects if dp['standard_name'].endswith(rdop_standard_name) and dp["dp_id"] != rdp_key][0]
                                if matching_dp in person_keys:
                                    
                                    reps[section_count][matching_dp] = self.sdprecords[section][nkey][matching_dp]
                                    rdps_to_remove.append(matching_dp)
                                    increase_count = True
                        if increase_count:
                            reps[section_count]["is_representative"] = True
                            reps[section_count]["title_as_appeared"] = self.sdprecords[section][nkey][rep_rdp["dp_id"]]
                            reps[section_count]["uuid"] = uuid.uuid4()
                            self.sdprecords[section][nkey]["representative_id"] = reps[section_count]["uuid"]
                            section_count = section_count + 1
                        for r in rdps_to_remove:
                            del self.sdprecords[section][nkey][r]
                if len(reps) > 0:
                    final = defaultdict(rec_dd)
                    final.update({**self.sdprecords[section], **reps})
                    self.sdprecords[section] = final
    
    def restructureDataforDB(self, breadcrumbs: list[str], dict_ = None,):
        #if dict_ is None:
        #    self.esc = self.extracted_sections
        target = dict_ if dict_ is not None else self.extracted_sections
        targetKeys = target.keys()
        for key in targetKeys:
            breadcrumbs.append(key)
            if key == "headers":
                self.paths_to_delete.append(breadcrumbs.copy())
                breadcrumbs.pop()
                continue
            if key.__contains__("statement"):
                for tdpm in self.table_dp_mapping.keys():
                    if data_mapping['dp_id'] in self.table_dp_mapping[tdpm]:
                        table = tdpm
                        break
                dps = self.map_statement(key)
                self.sdprecords[table] = {**self.sdprecords[table], **dps}
            if type(target[key]) is dict or type(target[key]) is defaultdict:
                if "state" in list(target[key].keys()):
                    entity_sepd = key.split(SEGSEP)
                    table = self.getSectionTable(self.table_mapping, entity_sepd[0])
                    self.ownerrecords.append(ListEntity(type_=entity_sepd[0], name=entity_sepd[-1], table=table, hash=shortuuid.ShortUUID().random(length=28), form_id=self.form_id))
                self.restructureDataforDB(breadcrumbs=breadcrumbs, dict_= target[key])
                #self.paths_to_delete.append(breadcrumbs.copy())
            else:
                parent_section = None
                owner_id = None
                owner_table = None
                owner_hash = None
                owner_type = None
                owner_name = None
                list_entry_id = None
                data_mapping = None
                if len(breadcrumbs) > 1:
                    item = self.replacePredefinedSuffixes(breadcrumbs[-2])
                    isItem = regex.match(r'item[0-9]{1,4}', item)
                if isItem is not None:
                    list_entry_id = int(self.replacePredefinedSuffixes(breadcrumbs[-2]).replace("item", ''))
                    bcwithoutSuffixes = [self.replacePredefinedSuffixes(d) for d in breadcrumbs if self.replacePredefinedSuffixes(d) != ""]
                    parent_section = self.replacePredefinedSuffixes(bcwithoutSuffixes[-3])
                else:
                    parent_section = self.replacePredefinedSuffixes(breadcrumbs[-2])
                
                field_name_in_form = self.replacePredefinedSuffixes(key)
                base_table = self.getSectionTable(self.table_mapping, breadcrumbs[0].split(SEGSEP)[0])
                base_table = regex.regex.sub(r'(?<=_\d{1,})_\d{1,}$', '', base_table)
                if breadcrumbs[0].__contains__(SEGSEP):
                    [owner_type, owner_name] = [breadcrumbs[0].split(SEGSEP)[0], breadcrumbs[0].split(SEGSEP)[-1]]
                    for id, owner in enumerate(self.ownerrecords):
                        if owner.type == owner_type and owner.name == owner_name:
                            owner_id = id
                            owner_table = owner.table
                            owner_hash = owner.hash
                            break 
                    for mapping in self.data_mapping:
                        if self.fieldExists(mapping['field_name'], field_name_in_form) and (self.extractedExistsinDefined(mapping['section'], parent_section) and mapping['dp_id'] in self.getTableDPs(base_table)):
                            data_mapping = mapping
                            break
                        elif self.fieldExists(mapping['field_name'], field_name_in_form) and self.extractedExistsinDefined(mapping['section'], owner_type):
                            data_mapping = mapping
                            break

                else:
                    for mapping in self.data_mapping:
                        if self.fieldExists(mapping['field_name'], field_name_in_form) and self.extractedExistsinDefined(mapping['section'], parent_section):
                            data_mapping = mapping
                            break
                                
                if data_mapping is not None:
                    table = None
                    if data_mapping['dp_id'] in ['DP_503', 'DP_569', 'DP_680', "DP_524", "DP_527", "DP_528", "DP_526", "DP_536", "DP_537", "DP_540", "DP_541", "DP_542", "DP_529", "DP_533", "DP_530", "DP_534", "DP_531", "DP_532", "DP_535", "DP_564"]:
                        table = self.getSectionTable(self.table_mapping, owner_type) if owner_type is not None else self.getSectionTable(self.table_mapping, breadcrumbs[0].split(SEGSEP)[0])
                    else:
                        for tdpm in self.table_dp_mapping.keys():
                            if data_mapping['dp_id'] in self.table_dp_mapping[tdpm]:
                                table = tdpm
                                break

                    owner_id_std = owner_id
                    ouuid = False
                    if table is not None and owner_id is not None and owner_table is not None and table != owner_table:
                        if regex.regex.search(r'(?<=_\d{1,})_\d{1,}$', table) != None and regex.regex.search(r'(?<=_\d{1,})_\d{1,}$', owner_table) != None:
                            owner_id_std = owner_table+"_standard"+"##"+str(owner_hash)
                            owner_id = owner_table+"##"+str(owner_hash)
                    
                        elif regex.regex.search(r'(?<=_\d{1,})_\d{1,}$', table) == None and regex.regex.search(r'(?<=_\d{1,})_\d{1,}$', owner_table) == None:
                            owner_id_std = None
                            owner_id = None
                    
                    if table is not None and owner_id is not None and owner_table is not None and table == owner_table:
                        ouuid = owner_hash
                        
                    if table is not None and owner_id is not None and list_entry_id is not None:
                        self.sdprecords[table][owner_id][list_entry_id][data_mapping['dp_id']] = target[key]
                        self.sdprecordsstandard[table+"_standard"][owner_id_std][list_entry_id][data_mapping['standard_name']] = target[key]
                        
                        if ouuid:
                            self.sdprecords[table][owner_id][list_entry_id]['uuid'] = ouuid
                            self.sdprecordsstandard[table+"_standard"][owner_id_std][list_entry_id]['uuid'] = ouuid

                        self.paths_to_delete.append(breadcrumbs.copy())
                    elif table is not None and owner_id is not None:
                        self.sdprecords[table][owner_id][data_mapping['dp_id']] = target[key]
                        self.sdprecordsstandard[table+"_standard"][owner_id_std][data_mapping['standard_name']] = target[key]
                        
                        if ouuid:
                            tap = regex.sub(r".*:\s*", '', owner_name)
                            tap = regex.regex.sub(r'\s+(?=\w\s+)', '', tap).strip()
                            newlinep = regex.compile(r".*"+ r'(' + '|'.join([ e.replace(' ', r'\s*') for e in self.table_mapping[table]]) + r')' +r"\s*", flags=regex.IGNORECASE | regex.MULTILINE)
                            tap = regex.sub(newlinep, '', tap)
                            # map auditors mandate type from section name
                            for (k, v) in self.tables_to_map_section_names_to_dps.items():
                                if table.lower().__contains__(k.lower()) and data_mapping['dp_id'] != v and not self.sdprecords[table][owner_id].keys().__contains__(v):
                                    self.sdprecords[table][owner_id][v] = self.normalizeString(breadcrumbs[0].split(SEGSEP)[0])
                            # craft title_as_appeared for moral persons to be identical to T3 
                            for k in self.sdprecords[table][owner_id].keys():
                                if k in self.rcs_dps and not tap.startswith(self.sdprecords[table][owner_id][k]):
                                    tap = self.sdprecords[table][owner_id][k] + " - " + tap
                            self.sdprecords[table][owner_id]['title_as_appeared'] = tap
                            self.sdprecords[table][owner_id]['uuid'] = ouuid
                            self.sdprecordsstandard[table+"_standard"][owner_id_std]['uuid'] = ouuid
                        
                        self.paths_to_delete.append(breadcrumbs.copy())
                    elif table is not None:
                        self.sdprecords[table][data_mapping['dp_id']] = target[key]
                        self.sdprecordsstandard[table+"_standard"][data_mapping['standard_name']] = target[key],
                        
                        if ouuid:
                            self.sdprecords[table]['uuid'] = ouuid
                            self.sdprecordsstandard[table+"_standard"]['uuid'] = ouuid
                            
                        self.paths_to_delete.append(breadcrumbs.copy())
            breadcrumbs.pop()
        return

class singularDP:
    def __init__(self, dp_id, field_name, field_value, name_in_form, parent_section = None, form_id = None, owner_id = None, list_entry_id = None):
        self.dp_id = dp_id
        self.field_name = field_name
        self.field_value = field_value
        self.name_in_form = name_in_form
        self.parent_section = parent_section
        self.form_id = form_id
        self.owner_id = owner_id
        self.list_entry_id = list_entry_id

class ListEntity:
    def __init__(self, type_, name, table, hash, form_id = None,):
        self.type: str = type_
        self.form_id: str = form_id
        self.name: str = name
        self.table = table
        self.hash = hash
