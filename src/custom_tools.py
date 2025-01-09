from langchain_core.tools import InjectedToolArg, tool
from langchain.tools.retriever import create_retriever_tool
import pyvo as vo
from pprint import pprint
from pyvo import registry
from typing import Dict, List, Optional
from pydantic import StringConstraints
from typing_extensions import Annotated
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.time import Time
from .data_types import RegistryConstraints, RegistryResponse, SIAResponse, VoImageResource, VoResource, VoImage, VoToolResponse, srvs, wavebands, ISO8601_date, img_types
from .retriever import retriever
import random

@tool(args_schema=RegistryConstraints)
def get_registry(words: list[str], 
                 service: Optional[srvs] = None, 
                 waveband: Optional[list[wavebands]] = None, 
                 author: Optional[str] = None, 
                 ivoid: Optional[str] = None, 
                 temporal: Optional[Annotated[str, StringConstraints(pattern=ISO8601_date)]] = None) -> RegistryResponse:
   """This function takes in contraints that are passed to a query against the virtual Observatory's Registry, a collection of metadata records of the resources in the VO. 
   The query returns a set of VO registry resources whose files and content match the given constraints. 
   If you are asked to search the registry, to provide "all data related to" specific objects or type of objects, 
   or if the request of the user doesn't specify from which service should you pull this information, use this tool.
   
   return:
      A JSON serializable object that contains a semnatic response for the Agent to read and pure data that is given straight to the user.
   """

   args = {"keywords":words if words[0] != "None" else None, "servicetype":service, "waveband":waveband, "author":author, "ivoid":ivoid, "temporal":Time(temporal) if temporal is not None else None}

   print(f"Constraints para registry.search: {args}")

   resources = registry.search(**{k: v for k, v in args.items() if v is not None}, maxrec=300)
   res_table = resources.to_table()['res_title','res_description','content_level','waveband','created','updated']

   print(f"Llamada a get_registry...\nCantidad resources encontrados: ({len(res_table)})")###
   if len(res_table) == 0:
      return RegistryResponse(semantic_response = f"No available resources could match the given constraints: {args}")

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

   print(f"Cantidad de resources a enviar al usuario: ({len(votable)})")###

   response = RegistryResponse(
      semantic_response = f"{len(votable)} resources where found in the query to the registry.", 
      data_response = votable
   )
   return response

@tool(parse_docstring=True)
def get_img(astro_object: str) -> VoToolResponse:
   """Use this function when the user asks to be shown an (any) image of a specific astronomical object. This function returns a random image that is realted to said astronomical object. 
   Do not confuse this function with the 'query_sia' function, which is more complete and returns a set of resources of graphical and FITS data.

   Args:
      astro_object: The name of the astronomical object or event the user requests an image of.

   return:
      A JSON serializable object that contains a semantic response for the Agent to read and an object that contains the properties of the image, as well as the link.
   """

   print("Llamada a get_img...")

   reg_results = registry.search(servicetype="sia")
   votable = reg_results.to_table()
   urls = votable["access_urls"]
   random.shuffle(urls)
   
   try:
      pos = SkyCoord.from_name(astro_object)
   except:
      return VoToolResponse(semantic_response = f"Sky Coord was not able to find coordinates for the name {astro_object}.")
   size = Quantity(0.5,'deg')

   for ref in urls:
      try:
         sia_service = vo.dal.SIAService(ref)
         sia_results = sia_service.search(pos=pos, size=size, format="graphic")
         if len(sia_results) == 0:
            print(f"No hay imagenes para '{astro_object}' en {ref}")
         else:            
            resource = sia_results.getrecord(random.randint(0, len(sia_results)))
            response = VoToolResponse(
               semantic_response = f"Image of name {resource.title} on format {resource.format} and size {resource.filesize} found in {resource.acref}.",
               data_response = VoImage(
                  title = str(resource.title),
                  format = str(resource.format),
                  filesize = resource.filesize,
                  url = str(resource.acref)
               )
            )
            return response
         
      except Exception as e:
         print(f"Error en la query a sia_service: {e}")
         continue
      
   return VoToolResponse(semantic_response = f"No available image data could match the given constraints:\nposition: {pos}\narea size: {size}")


@tool(parse_docstring=True)
def query_sia(position: str, img_type: Optional[img_types] = "all", unit: Optional[str] = 'deg', area: Optional[float] = 0.5, url: Optional[str] = None) -> SIAResponse:
   """This function is for retriving image resources. Execute it when the user requests images (plural), visual or graphic data.
   More specifically, this function queries a specified resource with the Simple Image Access protocol to retrieve an image or FITS file whose data was obtained by a telescope of a region or an object in space. It takes in the position and size of an area in the skydome (what we see when we look into space) and the url of a resource that may contain records and images of this area.
   
   Args:
      position: The particular name of an astronomical object. A function will parse it into sky coordinates. It is crucial that the name is a 'particular' name and not a generic 'type of object'.
      img_type: The time of graphical data to search for. It can one of: 'images', 'fits', 'all'
      area: Size of the area in which records are to be searched
      unit: Unit of the amount described by the area arg
      url: (optional) url of the service to query

   return:
      A JSON serializable object that contains a semantic response for the Agent to read and a list of resources to get data from. The later is only ever shown to the user.
   """

   print("Llamada a query_sia...")

   if url is None:
      reg_results = registry.search(servicetype="sia")
      votable = reg_results.to_table()
      urls = votable["access_urls"]
      random.shuffle(urls)
   else:
      urls = [url]

   match img_type:
      case "images":
         img_types = ["image/jpeg","image/png"]
      case "fits":
         img_types = ["image/fits"]
      case "all":
         img_types = "all"
   
   try:
      pos = SkyCoord.from_name(position)
   except:
      return SIAResponse(semantic_response = f"Sky Coord was not able to find coordinates for the name {position}.")
   size = Quantity(area, unit)

   for ref in urls:
      try:
         sia_service = vo.dal.SIAService(ref)
         sia_results = sia_service.search(pos=pos, size=size, format=img_types)
         if len(sia_results) == 0:
            print(f"No hay imagenes para '{position}' en {ref}")
         else:
            response = SIAResponse(
               semantic_response = f"Se encontraron {len(sia_results)} imágenes en {ref}.",
               data_response = []
            )
            count = 0
            for rec in sia_results:
               if count > 100:
                  break
               response.data_response.append(
                  VoImageResource(
                     title = str(rec.title),
                     format = str(rec.format),
                     filesize = rec.filesize,
                     url = str(rec.acref),
                     instrument = str(rec.instr),
                     position = rec.pos.to_string(),
                     bandpass = str(rec.bandpass_id) if rec.bandpass_id is not None else None,
                     bandpass_reference = f"{rec.bandpass_refvalue} [{rec.bandpass_unit}]" if rec.bandpass_id is not None else None,
                     bandpass_range = f"({rec.bandpass_lolimit}, {rec.bandpass_hilimit})" if rec.bandpass_id is not None else None
                  )
               )
               count+=1
            return response
         
      except Exception as e:
         print(f"Error en la query a sia_service: {e}")
         continue
      
   return SIAResponse(semantic_response = f"No available image data could match the given constraints:\nposition: {pos}\narea size: {size}")


@tool(parse_docstring=True)
def query_ssa(position: str, diameter: Optional[float]=0.1, band: Optional[tuple] = None, time: Optional[tuple] = None, url: Optional[str] = None) -> VoToolResponse:
   """This tool queries a Simple Spectral Access service to retrieve spectra-related data from the VO. Use it when the user asks specifically for 'spectral data of...' and if they ask for a particular object, and not a type of object (this is very important). 
   If a url is not provided it will query the registry to find a service that matches the constraints passed in as arguments. If a url is provided or a matching service 
   was found, it will query it and return a dictionary with relevant data of the resource.

   Args:
      position: The particular name of an astronomical object. A function will parse it into sky coordinates. It is crucial that the name is a 'particular' name and not a generic 'type of object'.
      diameter: The diameter of the circular region around position in which to search, assuming icrs decimal degrees
      band: (optional) The bandwidth range the data needs to match, assuming meters
      time: (optional) The datetime range  the data needs to match
      url: (optional) The url of the resource to be queried. If none provided, the position will be used as keywords to search the registry
      
   return:
      A JSON serializable object that contains a semantic response for the Agent to read and a list of resources to get data from. The later is only ever shown to the user.
   """
   try:
      sky_pos = SkyCoord.from_name(position)
   except:
      return VoToolResponse(semantic_response = f"Sky Coord was not able to find coordinates for the name {position}.")

   args = {"pos":sky_pos, "diameter":diameter, "band":band, "time":time}

   if url is None:
      ssa_services = vo.regsearch(servicetype='ssa')
      votable = ssa_services.to_table()
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
   This tool is similar to the tool 'get_registry', the difference is that this tool queries services recovered from the registry via Simple Cone Search. This means that any resource that has data whose position lies within a cone described by the position and radius given to this tool as arguments is returned in a data table that is structured as a python dictionary.

   Args:
      position: The 'particular' name of the astronomical object that represents the center of the circular search region where the resource data must lie within to match the constraint. This value must not be a 'type' of astronomical object, but the specific name. The area of this region is described by the radius.
      radius: Radius of the circular search region.
      url: (optional) The url of the resource to be queried. If none provided, the position will be used as keywords to search the registry.
      
   return:
      A JSON serializable object that contains a semantic response for the Agent to read and a list of resources to get data from. The later is only ever shown to the user.
   """

   try:
      sky_pos = SkyCoord.from_name(position)
   except:
      return VoToolResponse(semantic_response = f"Sky Coord was not able to find coordinates for the name {position}.")
   args = {"pos":sky_pos, "radius":radius}

   if url is None:
      scs_services = vo.regsearch(servicetype='conesearch')
      votable = scs_services.to_table()
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
def query_sla(wavelength: tuple, unit: Optional[str]='meter', url: Optional[str] = None) -> VoToolResponse:
   """
   This tool takes in a wavelength range and queries the VO using the Simple Line Access protocol to recover data of 
   observations of elements in celestial bodies and space.

   Args:
      wavelength: Pair of floats that describe a wavelength range. The results of the query to the service lie within this range.
      unit: The unit of the quantities that describe the wavelenght range.
      url: (optional) The url of the resource to be queried. If none provided, the position will be used as keywords to search the registry.
      
   return:
      A JSON serializable object that contains a semantic response for the Agent to read and a list of resources to get data from. The later is only ever shown to the user.
   """

   waverange = Quantity(wavelength, unit=unit)
   
   if url is None:
      sla_services = vo.regsearch(servicetype='line')
      votable = sla_services.to_table()
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

retriever_tool = create_retriever_tool(
   retriever,
   "retrieve_scientific_documents",
   "Search for (mainly) Virtual Observatory realted information, either to compliment your answers or to better understand what the user is asking you to search for. This tool is not to be confused with the tools to search the Virtual Observatory itself"
)

# print(get_registry.invoke({"words": "any", "service": "any", "name": "any"}))
# print(query_sia.name)
# print(get_registry.description)
# print(get_registry.args)