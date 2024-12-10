from langchain_core.tools import InjectedToolArg, tool
import pyvo as vo
from pprint import pprint
from pyvo import registry
from typing import List, Optional
from pydantic import StringConstraints
from typing_extensions import Annotated
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from .data_types import RegistryConstraints, RegistryResponse, SIAResponse, VoResource, VoImage, VoToolResponse, srvs, wavebands


@tool(args_schema=RegistryConstraints)
def get_registry(words: list[str], 
                 service: Optional[srvs] = None, 
                 waveband: Optional[list[wavebands]] = None, 
                 author: Optional[str] = None, 
                 ivoid: Optional[str] = None, 
                 temporal: Optional[Annotated[str, StringConstraints(pattern=r'[0-9]{4}-[0-9]{2}-[0-9]{2}$')]] = None) -> RegistryResponse:
   """This function takes in contraints that are passed to a query against the virtual Observatory's Registry, a collection of metadata records of the resources in the VO. 
   The query returns a set of VO registry resources whose files and content match the given constraints.
   
   Returns:
      list[VoResource]: List of VoResources, dictionary like objects that describe each of the resources found in the query. The data they provide are: 
      resource title, resource description, the knowledge level of the resource's intended audience, the waveband of the resources, the created and last 
      updated dates of the resources, and the url's to access them.
   """

   args = {"keywords":words if words[0] != "None" else None, "servicetype":service, "waveband":waveband, "author":author, "ivoid":ivoid, "temporal":temporal}

   print(f"Constraints para registry.search: {args}")

   resources = registry.search(**{k: v for k, v in args.items() if v is not None}, maxrec=300)
   res_table = resources.to_table()['res_title','res_description','content_level','waveband','created','updated']

   print(f"Llamada a get_registry...\nCantidad resources encontrados: ({len(res_table)})")###

   votable = []
   count = 20

   print(f"Limite de resultados: {count}")

   for i in range(0, len(res_table)):
      if i > count:
         break
      # descriptions[res.short_name] = str(res.res_title) + "\n" + str(res.res_description)
      table_res = res_table[i]
      modes = list(resources.getrecord(i).access_modes())
      urls = resources.getrecord(i).access_url
      votable.append(VoResource(**table_res, access_modes=modes, access_urls=urls))# access_modes()
      count-=1

   print(f"Cantidad de resources a enviar al usuario: ({len(votable)})")###

   response = RegistryResponse(
      semantic_response = f"{len(votable)} resources where found in the query to the registry.", 
      data_response = votable
   )
   return response

@tool(parse_docstring=True)
def query_sia(position: str, unit: Optional[str] = 'deg', area: Optional[float] = 0.5, url: Optional[str] = None) -> SIAResponse:
   """This function is for retriving images. Execute it when the user requests an image.
   More specifically, this function queries a specified resource with the Simple Image Access protocol to retrieve an image taken by a telescope of a region or an object in space. 
   It takes in the position and size of an area in the skydome (what we see when we look into space) and the url of a resource that may contain records and images of this area.
   The return is a dictionary. The keys correspond to the url where the image may be found and the values are the name, file type and size of the image.
   
   Args:
      position: The name of an astronomical object. A function will parse it into sky coordinates
      area: Size of the area in which records are to be searched
      unit: Unit of the amount described by the area arg
      url: (optional) url of the service to query
   """

   print("Llamada a query_sia...")

   # pos = SkyCoord.from_name('Eta Carina')
   # size = Quantity(0.5, unit="deg")
   # sia_service = vo.dal.SIAService('https://cda.harvard.edu/cxcsiap/queryImages?')
   if url is None:
      # votable = get_registry([position], service='sia')###<---- USE .INVOKE
      reg_results = registry.search(servicetype="sia")
      votable = reg_results.to_table()
      if len(votable) == 0:
         return SIAResponse(semantic_response = f"No available resources could match the given constraints:\nwords: '{position}'\nservice: 'sia'")
      urls = votable["access_urls"]
   else:
      urls = [url]
   
   try:
      pos = SkyCoord.from_name(position)
   except:
      return SIAResponse(semantic_response = f"Sky Coord was not able to find coordinates for the name {position}.")
   size = Quantity(area, unit)

   for ref in urls:
      try:
         sia_service = vo.dal.SIAService(ref)
         sia_results = sia_service.search(pos=pos, size=size, format="graphic")
         if len(sia_results) == 0:
            print(f"No hay imagenes para '{position}' en {ref}")
         else:
            for rec in sia_results:
               reply = f"Imagen encontrada en {ref}: {rec.title, rec.format, rec.filesize}"
               print(reply)
               response = SIAResponse(
                  semantic_response = reply, 
                  data_response = VoImage(title=rec.title, link=rec.acref, format=rec.format, filesize=rec.filesize)
               )
               return response
            
      except Exception as e:
         print(f"Error en la query a sia_service: {e}")
         continue
      
   return SIAResponse(semantic_response = f"No available image data could match the given constraints:\nposition: {pos}\narea size: {size}\nformat: 'graphics'")

@tool(parse_docstring=True)
def query_ssa(position: str, diameter: Optional[float]=0.1, band: Optional[tuple] = None, time: Optional[tuple] = None, url: Optional[str] = None) -> VoToolResponse:
   """This tool queries a Simple Spectral Access service to retrieve spectra-related data from the VO. Use it when the user asks specifically for 'spectral data of...'. 
   If a url is not provided it will query the registry to find a service that matches the constraints passed in as arguments. If a url is provided or a matching service 
   was found, it will query it and return a dictionary with relevant data of the resource.

   Args:
      position: The name of an astronomical object. A function will parse it into sky coordinates
      diameter: The diameter of the circular region around position in which to search, assuming icrs decimal degrees
      band: (optional) The bandwidth range the data needs to match, assuming meters
      time: (optional) The datetime range  the data needs to match
      url: (optional) The url of the resource to be queried. If none provided, the position will be used as keywords to search the registry
   """
   try:
      sky_pos = SkyCoord.from_name(position)
   except:
      return VoToolResponse(semantic_response = f"Sky Coord was not able to find coordinates for the name {position}.")

   args = {"pos":sky_pos, "diameter":diameter, "band":band, "time":time}

   if url is None:
      ssa_services = vo.regsearch(servicetype='ssa')
      votable = ssa_services.to_table()
      if len(votable) == 0:
         return VoToolResponse(semantic_response = f"No available resources could match the given constraints:\nwords: '{position}'\nservice: 'ssa'")
      urls = votable["access_urls"]
   else:
      urls = [url]

   for ref in urls:
      try:
         ssa_service = vo.dal.SSAService(ref)
         ssa_results = ssa_service.search(**{k: v for k, v in args.items() if v is not None})
         if len(ssa_results) == 0:
            print(f"No hay data para '{position}' en {ref}")
         else:
            ssa_table = ssa_results.to_table()
            colnames = ssa_table.colnames
            rows = ssa_table.as_array()
            table_dicts = []
            lim = 0
            for row in rows:
               if lim >= 20:
                  break
               str_row = [str(x) for x in row]
               new_dict = dict( tuple( zip(colnames,str_row) ) )
               table_dicts.append(new_dict)
               lim+=1

            response = VoToolResponse(
               semantic_response = f"{len(table_dicts)} resources where found in the Simple Spectral Access query to {ref}.",
               data_response = table_dicts
            )
            return response
         
      except Exception as e:
         print(f"Error en query_ssa: {e}")
         continue
   return VoToolResponse(semantic_response = f"No available image data could match the given constraints:\n{args}.")


@tool(parse_docstring=True)
def query_scs(position: str, radius: Optional[float]=0.1, url: Optional[str] = None) -> VoToolResponse:
   """
   This tool is similar to the tool 'get_registry', the difference is that this tool queries services recovered from 
   the registry via Simple Cone Search. This means that any resource that has data whose position lies within a cone 
   described by the position and radius given to this tool as arguments is returned in a data table that is structured 
   as a python dictionary.

   Args:
      position: The center of the circular search region where the resource data must lie within to match the constraint. The area of this region is described by the radius.
      radius: Radius of the circular search region.
      url: (optional) The url of the resource to be queried. If none provided, the position will be used as keywords to search the registry.
   """

   try:
      sky_pos = SkyCoord.from_name(position)
   except:
      return VoToolResponse(semantic_response = f"Sky Coord was not able to find coordinates for the name {position}.")
   args = {"pos":sky_pos, "radius":radius}

   if url is None:
      scs_services = vo.regsearch(servicetype='conesearch')
      votable = scs_services.to_table()
      if len(votable) == 0:
         return VoToolResponse(semantic_response = f"No available resources could match the given constraints:\nwords: '{position}'\nservice: 'ssa'")
      urls = votable["access_urls"]
   else:
      urls = [url]

   for ref in urls:
      try:
         scs_service = vo.dal.SSAService(ref)
         scs_results = scs_service.search(**{k: v for k, v in args.items() if v is not None})
         if len(scs_results) == 0:
            print(f"No hay data para '{position}' en {ref}")
         else:
            scs_table = scs_results.to_table()
            colnames = scs_table.colnames
            rows = scs_table.as_array()
            table_dicts = []
            lim = 0
            for row in rows:
               if lim >= 20:
                  break
               str_row = [str(x) for x in row]
               new_dict = dict( tuple( zip(colnames,str_row) ) )
               table_dicts.append(new_dict)
               lim+=1
            
            response = VoToolResponse(
               semantic_response = f"{len(table_dicts)} resources where found in the Simple Cone Search query to {ref}.",
               data_response = table_dicts
            )
            return response
         
      except Exception as e:
         print(f"Error en query_scs: {e}")
   return VoToolResponse(semantic_response = f"No available cone search could match the given constraints:\n{args}.")


@tool(parse_docstring=True)
def query_sla(wavelength: List[float], unit: Optional[str]='meter', url: Optional[str] = None) -> VoToolResponse:
   """
   This tool takes in a bandwidth range and queries the VO using the Simple Line Access protocol to recover data of 
   observations of elements in celestial bodies and space.

   Args:
      wavelength: Pair of floats that describe a wavelength range, . The results of the query to the serviec lie within this range.
      url: (optional) The url of the resource to be queried. If none provided, the position will be used as keywords to search the registry.
   """

   waverange = Quantity(wavelength, unit=unit)
   
   if url is None:
      sla_services = vo.regsearch(servicetype='line')
      votable = sla_services.to_table()
      if len(votable) == 0:
         return VoToolResponse(semantic_response = f"No available resources could match the given constraints:\nbandwith range = {waverange}\nservice: 'sla'")
      urls = votable["access_urls"]
   else:
      urls = [url]

   for ref in urls:
      try:
         sla_service = vo.dal.SLAService(ref)
         sla_results = sla_service.search(wavelength=waverange)
         if len(sla_results) == 0:
            print(f"No hay data para el rango '{waverange}' en {ref}")
         else:
            sla_table = sla_results.to_table()
            colnames = sla_table.colnames
            rows = sla_table.as_array()
            table_dicts = []
            lim = 0
            for row in rows:
               if lim >= 20:
                  break
               str_row = [str(x) for x in row]
               new_dict = dict( tuple( zip(colnames,str_row) ) )
               table_dicts.append(new_dict)
               lim+=1

            response = VoToolResponse(
               semantic_response = f"{len(table_dicts)} resources where found in the Simple Spectral Access query to {ref}.",
               data_response = table_dicts
            )
            return response
                  
      except Exception as e:
         print(f"Error en query_scs: {e}")

   return VoToolResponse(semantic_response = f"No available cone search could match the given constraints:\nwavelength = {waverange}.")


# @tool()
# def query_tap():
#    """
#    """
#    return

# print(get_registry.invoke({"words": "any", "service": "any", "name": "any"}))
# print(query_sia.name)
# print(get_registry.description)
# print(get_registry.args)