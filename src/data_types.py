from pydantic import BaseModel, Field, StringConstraints
from astropy.time import Time
from typing_extensions import Annotated, TypedDict
from typing import Literal, Optional, Any
from langchain_core.tools import InjectedToolArg

srvs = Literal['sia', 'sia1', 'sia2', 'ssa', 'ssap', 'scs', 'conesearch', 'line', 'tap', 'table']

wavebands = Literal['EUV', 'Gamma-ray', 'Infrared', 'Millimeter', 'Neutrino', 'Optical', 'Photon', 'Radio', 'UV', 'X-ray']

class VoToolResponse(BaseModel):
   semantic_response: str
   data_response: Optional[list[dict[str,str]]] = Field(default=None)

class VoResource(TypedDict):
   res_title: str
   res_description: str
   content_level: str
   waveband: str
   created: str
   updated: str
   access_urls: str
   access_modes: list[str]

class VoImage(TypedDict):
   title: str = Field(description="Title of the image")
   link: str = Field(description="Link to the image")
   format: str = Field(description="Format fo the image")
   filesize: Optional[float] = Field(default=None, description="The filesize o the image in Bytes")

class RegistryResponse(VoToolResponse):
   data_response: Optional[list[VoResource]] = Field(default=None)

class SIAResponse(VoToolResponse):
   data_response: Optional[VoImage] = Field(default=None)
   
class RegistryConstraints(BaseModel):
   words: list[str] = Field(description="Keywords that should be found in the resources of the registry")
   service: Optional[srvs] = Field(default=None, description="Service type that the resources found should serve. Could be any of: 'sia', 'sia1', 'sia2', 'ssa', 'ssap', 'scs', 'conesearch', 'line', 'tap', 'table'")
   waveband: Optional[list[wavebands]] = Field(default=None, description="Waveband in which the resources data lies within")
   author: Optional[str] = Field(default=None, description="Author of (some of) the data found in the resources")
   ivoid: Optional[str] = Field(default=None, description="Exact if of the resource to be found")
   temporal: Optional[Annotated[str, StringConstraints(pattern=r'[0-9]{4}-[0-9]{2}-[0-9]{2}$')]] = Field(default=None, description="Time in which the resource was publiches or last updated")

class RegistryResponseParser(BaseModel):
   """If the last tool you called is get_registry, respond to the user with this"""

   text_answer: str = Field(description="A concise answer to the user query. This message exist to present the user with the answer to their query, which will be a set of resources in a table. An example woudl be 'I have found 20 sources of information for your quey on asteroids'.")
   data_table: list[VoResource] = Field(description="A dictionary like object containing al data present in tha data table")

class ImageResponseParser(BaseModel):
   """If the last tool you called is query_sia, respond to the user with this"""
   text_answer: str = Field(description="A short text answer to the user to acompanny the image (result of the query)")
   vo_image: VoImage = Field(description="A dictionary representing the image returned bby the user query.")