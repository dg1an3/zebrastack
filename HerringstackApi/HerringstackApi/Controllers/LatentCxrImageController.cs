using HerringstackApi.Abstractions;
using HerringstackApi.Services;
using Microsoft.AspNetCore.Mvc;

// For more information on enabling Web API for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace HerringstackApi.Controllers;

[Route("api/[controller]")]
[ApiController]
public class LatentCxrImageController : ControllerBase
{
    private readonly ICxr8DataManager _dataManager;

    public LatentCxrImageController(ICxr8DataManager dataManager)
    {
        _dataManager = dataManager;
    }

    // GET: api/<LatentCxrImageController>
    [HttpGet]
    public async Task<IEnumerable<Subject>> Get()
    {
        var subjectItems = await _dataManager.GetSubjectItemsAsync();
        return subjectItems;
    }

    // GET api/<LatentCxrImageController>/5
    [HttpGet("{id}")]
    public string Get(int id)
    {
        return "value";
    }

    // POST api/<LatentCxrImageController>
    [HttpPost]
    public void Post([FromBody] string value)
    {
    }

    // PUT api/<LatentCxrImageController>/5
    [HttpPut("{id}")]
    public void Put(int id, [FromBody] string value)
    {
    }
    
    // DELETE api/<LatentCxrImageController>/5
    [HttpDelete("{id}")]
    public void Delete(int id)
    {
    }
}
